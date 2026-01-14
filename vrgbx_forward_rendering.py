import os
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
import cv2
import imageio.v2 as iio
from PIL import Image

from safetensors.torch import load_file
from vrgbx import save_video
from vrgbx.pipelines.vrgbx_model import VRGBXPipeline, ModelConfig


# modality mapping
TYPE_ORDER = ["albedo", "normal", "material", "irradiance"]
TYPE_ID = {t: i for i, t in enumerate(TYPE_ORDER)}  # 0..3


def parse_args():
    p = argparse.ArgumentParser("RGBX->RGB video inference (local ckpt)")

    # inputs
    p.add_argument("--albedo_path", type=str, default="examples/input_intrinsics/Evermotion_Banquet_Albedo.mp4")
    p.add_argument("--normal_path", type=str, default="examples/input_intrinsics/Evermotion_Banquet_Normal.mp4")
    p.add_argument("--material_path", type=str, default="examples/input_intrinsics/Evermotion_Banquet_Material.mp4")
    p.add_argument("--irradiance_path", type=str, default="examples/input_intrinsics/Evermotion_Banquet_Irradiance.mp4")

    # reference image
    p.add_argument("--use_reference", action="store_true", help="Enable reference RGB conditioning. If set, --ref_rgb_path is required.")
    p.add_argument("--ref_rgb_path", type=str, default=None)

    # mixing
    p.add_argument("--edit_type", type=str, default=None, choices=TYPE_ORDER)
    p.add_argument("--edit_x_path", type=str, default=None)

    # video
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--fps", type=int, default=15)

    # output
    p.add_argument("--out_dir", type=str, default="output/forward_rendering")
    p.add_argument("--quality", type=int, default=5)
    p.add_argument("--save_mixed", action="store_true")

    # inference
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n_seeds", type=int, default=1)
    p.add_argument("--tiled", action="store_true")
    p.add_argument("--ref_cfg_scale", type=float, default=1.5)

    # pipeline
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--torch_dtype", type=str, default="bfloat16")

    # checkpoint
    p.add_argument("--ckpt_path", type=str, default='models/V-RGBX/vrgbx_forward_renderer.safetensors', help="Local .safetensors path")
    p.add_argument("--ckpt_name", type=str, default="X2RGB")

    return p.parse_args()


def build_pipe(device: str, torch_dtype: str):
    dtype = getattr(torch, torch_dtype)

    pipe = VRGBXPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu", skip_download=True),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", skip_download=True),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu", skip_download=True),
        ],
        rewrite_forward=True,
        change_dim=52,
        num_types=4,
    )
    return pipe


def ensure_hw(frame: np.ndarray, H: int, W: int) -> np.ndarray:
    h, w = frame.shape[:2]
    src_ar, tgt_ar = w / h, W / H

    if abs(src_ar - tgt_ar) > 1e-6:
        if src_ar < tgt_ar:
            nh = int(round(w / tgt_ar))
            frame = frame[(h - nh) // 2:(h - nh) // 2 + nh, :]
        else:
            nw = int(round(h * tgt_ar))
            frame = frame[:, (w - nw) // 2:(w - nw) // 2 + nw]

    frame = cv2.resize(frame, (W, H))
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    return frame[:, :, :3]


def read_video(path: Path, F: int, H: int, W: int):
    rdr = iio.get_reader(str(path))
    frames = []
    for i, f in enumerate(rdr):
        if i >= F:
            break
        frames.append(ensure_hw(f, H, W))
    rdr.close()

    if not frames:
        raise ValueError(f"No frames read from {path}")

    while len(frames) < F:
        frames.append(frames[-1])
    return frames


def gather_paths(args):
    paths = {}
    for t in TYPE_ORDER:
        p = getattr(args, f"{t}_path")
        if p:
            paths[t] = Path(p)
    if not paths:
        raise ValueError("No input modalities provided.")
    return paths


def intrinsic_video_sample(
    paths: Dict[str, Path],
    F: int, H: int, W: int,
    edit_type: Optional[str],
    edit_x_path: Optional[str],
):
    avail = [t for t in TYPE_ORDER if t in paths]
    clips = {t: read_video(paths[t], F, H, W) for t in avail}

    # edit_type can be None
    if edit_type is None:
        frames, type_ids = [], []
        for i in range(F):
            t = random.choice(avail)     
            frames.append(clips[t][i]) 
            type_ids.append(TYPE_ID[t])
        return frames, type_ids

    # original behavior when edit_type is provided
    if edit_type not in avail:
        raise ValueError(f"{edit_type} not provided. Available: {avail}")
    pool = [t for t in avail if t != edit_type]
    if not pool:
        raise ValueError("Need at least 2 modalities.")

    frames = [clips[edit_type][0]]
    type_ids = [TYPE_ID[edit_type]]

    for i in range(1, F):
        t = random.choice(pool)
        frames.append(clips[t][i])
        type_ids.append(TYPE_ID[t])

    # only override first frame content when both exist
    if edit_x_path is not None:
        img = iio.imread(edit_x_path)
        frames[0] = ensure_hw(img, H, W)

    return frames, type_ids



def load_input_image(path: Optional[str], H: int, W: int):
    if path is None:
        # dummy black image [H, W, 3]
        dummy = np.zeros((H, W, 3), dtype=np.uint8)
        return Image.fromarray(dummy, mode="RGB")

    return Image.open(path).convert("RGB").resize((W, H))



def main():
    print("start")
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # ---- reference gate ----
    if args.use_reference:
        if args.ref_rgb_path is None:
            raise ValueError("--use_reference is set but --ref_rgb_path is None")
        input_image0 = load_input_image(args.ref_rgb_path, args.height, args.width)
        rgb_dropout = 0.0
        ref_cfg_scale = args.ref_cfg_scale
    else:
        input_image0 = None
        rgb_dropout = 1.0
        ref_cfg_scale = -1.0   # explicitly disable ref conditioning

    H, W, F = args.height, args.width, args.num_frames
    out_dir = Path(args.out_dir) / f"{args.ckpt_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Load inputs...")
    paths = gather_paths(args)

    print("[2/4] Build pipeline...")
    pipe = build_pipe(args.device, args.torch_dtype)
    pipe.dit.load_state_dict(load_file(args.ckpt_path, device="cpu"))
    print(f"✅ [X2RGB] Loaded forward renderer weights from: {args.ckpt_path}")
    pipe.enable_vram_management()

    print("[3/4] Prepare mixed video...")
    intrinsic_video, frame_type_ids = intrinsic_video_sample(
        paths, F, H, W, args.edit_type, args.edit_x_path
    )
    input_image0 = load_input_image(args.ref_rgb_path, H, W)
    pairs = list(paths.items())
    channel_, path_ = pairs[0]
    channel_ = channel_[:1].upper() + channel_[1:]
    base = os.path.basename(path_).replace(f'_{channel_}.mp4', '')
    flag = 'w_ref' if args.use_reference else 'no_ref'

    if args.save_mixed:
        iio.mimwrite(
            out_dir / f"{base}_{flag}_intrinsic_video.mp4",
            [f.astype(np.uint8) for f in intrinsic_video],
            fps=args.fps,
        )

    print("[4/4] Run inference...")
    for _ in range(args.n_seeds):
        seed = args.seed or random.randint(1, 10**9)
        video = pipe(
            prompt="#rgb",
            rgbx_video=intrinsic_video,
            input_image=input_image0,
            frame_type_ids=frame_type_ids,
            seed=seed,
            tiled=args.tiled,
            num_frames=F,
            rgb_dropout=rgb_dropout,
            ref_cfg_scale=ref_cfg_scale,
        )
        out_path = out_dir / f"{base}_video_x2rgb_{flag}.mp4"
        save_video(video, str(out_path), fps=args.fps, quality=args.quality)
        print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
