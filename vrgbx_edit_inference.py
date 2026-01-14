import os
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import cv2
import imageio.v2 as iio
from PIL import Image
from safetensors.torch import load_file

from vrgbx import save_video, VideoData, load_state_dict
from vrgbx.pipelines.vrgbx_model import VRGBXPipeline, ModelConfig


TYPE_ORDER = ["albedo", "normal", "material", "irradiance"]
TYPE_ID = {t: i for i, t in enumerate(TYPE_ORDER)}


def load_video_slice(video_path: str, H: int, W: int, start: int, F: int):
    vd = VideoData(video_path, height=H, width=W)
    return [vd[i] for i in range(start, start + F)]


def ensure_hw(frame: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    center-crop to match target aspect ratio, then resize to (W, H), ensure 3-channel uint8
    """
    h, w = frame.shape[:2]
    src_ar, tgt_ar = w / h, W / H

    if abs(src_ar - tgt_ar) > 1e-6:
        if src_ar < tgt_ar:
            nh = int(round(w / tgt_ar))
            y0 = (h - nh) // 2
            frame = frame[y0 : y0 + nh, :]
        else:
            nw = int(round(h * tgt_ar))
            x0 = (w - nw) // 2
            frame = frame[:, x0 : x0 + nw]

    frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    return frame[:, :, :3].astype(np.uint8)


def read_video(path: Path, F: int, H: int, W: int) -> List[np.ndarray]:
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


def intrinsic_video_sample(paths: Dict[str, Path], F: int, H: int, W: int, edit_type: str, edit_x_path: Optional[str], drop_type=None):
    avail = [t for t in TYPE_ORDER if t in paths and t != drop_type]
    if edit_type not in avail:
        raise ValueError(f"{edit_type} not provided.")
    pool = [t for t in avail if t != edit_type]
    if not pool:
        raise ValueError("Need at least 2 modalities.")

    clips = {t: read_video(paths[t], F, H, W) for t in avail}

    frames = [clips[edit_type][0]]
    type_ids = [TYPE_ID[edit_type]]

    for i in range(1, F):
        t = random.choice(pool)
        frames.append(clips[t][i])
        type_ids.append(TYPE_ID[t])

    if edit_x_path:
        img = iio.imread(edit_x_path)
        frames[0] = ensure_hw(img, H, W)

    return frames, type_ids


def parse_args():
    p = argparse.ArgumentParser("RGB2X + X2RGB edit (random-mix forward rendering)")

    # inputs
    p.add_argument("--video_name", type=str, default=None, help="Video basename, e.g. Captured_PoolTable or Evermotion_Studio")
    p.add_argument("--video_path", type=str, default=None, help="Input RGB video path")
    p.add_argument("--ref_rgb_path", type=str, default=None, help="Reference RGB image (edited key frame rgb)")
    p.add_argument("--edit_type", type=str, default="albedo", choices=TYPE_ORDER, help="Which intrinsic is edited")
    p.add_argument("--task", type=str, default="texture", help="Editing task(e.g. texture, solid color, shadow, light color, normal, ...)")
    p.add_argument("--edit_x_path", type=str, default=None, help="Edited intrinsic image path (for frame-0)")

    # ckpts
    p.add_argument("--rgb2x_ckpt", type=str, default="models/V-RGBX/vrgbx_inverse_renderer.safetensors", help="Inverse renderer ckpt (.safetensors)")
    p.add_argument("--x2rgb_ckpt", type=str, default="models/V-RGBX/vrgbx_forward_renderer.safetensors", help="Forward renderer ckpt (.safetensors)")

    # video params
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--num_frames", type=int, default=49)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--quality", type=int, default=5)

    # device
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--torch_dtype", type=str, default="bfloat16")

    # pipeline dims
    p.add_argument("--rgb2x_change_dim", type=int, default=32)
    p.add_argument("--x2rgb_change_dim", type=int, default=52)

    # randomness / guidance
    p.add_argument("--seed", type=int, default=1, help="Global seed (affects mix + diffusion)")
    p.add_argument("--ref_cfg_scale", type=float, default=1.5)
    p.add_argument("--rgb_dropout", type=float, default=0.0)
    p.add_argument("--drop_type", type=str, default=None, choices=TYPE_ORDER, help="Modality to completely exclude from intrinsic_video composition (e.g. normal).")


    # debug saves
    p.add_argument("--save_rgb2x", default=True, help="Save rgb2x predicted modality videos")
    p.add_argument("--save_mixed", action="store_true", help="Save mixed interleaved intrinsic video")

    # output
    p.add_argument("--out_dir", type=str, default="output/intrinsic_edit") 
    

    return p.parse_args()

def build_pipe(device: str, torch_dtype: str, change_dim: int, num_types: int = -1):
    dtype = getattr(torch, torch_dtype)
    return VRGBXPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu", skip_download=True),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu", skip_download=True),
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu", skip_download=True),
        ],
        rewrite_forward=True,
        change_dim=change_dim,
        num_types=num_types,
    )

def main():
    args = parse_args()
    H, W, F = args.height, args.width, args.num_frames

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / ".tmp_rgb2x"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.video_name:
        base = args.video_name
        args.video_path = args.video_path or f"examples/input_videos/{base}.mp4"
        args.ref_rgb_path = args.ref_rgb_path or f"examples/edit_images/{base}_{args.task}_edit_ref.png"
        args.edit_x_path = args.edit_x_path or f"examples/edit_images/{base}_{args.task}_edit_x.png"
    else:
        base = os.path.basename(args.video_path).replace('.mp4', '')

    # -------- 1) RGB -> X (predict 4 modality videos) --------
    pipe_rgb2x = build_pipe(args.device, args.torch_dtype, change_dim=args.rgb2x_change_dim)
    pipe_rgb2x.dit.load_state_dict(load_state_dict(args.rgb2x_ckpt))
    print(f"✅ [RGB2X] Loaded inverse renderer weights from: {args.rgb2x_ckpt}")
    pipe_rgb2x.enable_vram_management()

    rgb_video = load_video_slice(args.video_path, H, W, args.start, F)

    pred_paths = {}
    for ch in TYPE_ORDER:
        p = tmp_dir / f"{base}_rgb2x_{args.start}_{args.start+F}_{ch}.mp4"
        if p.exists():
            print(f"[RGB2X] Found cached {ch}: {p}, skipping inference.")
        else:
            print(f"[RGB2X] Running inverse renderer for {ch}...")
            vid = pipe_rgb2x(prompt=f"#{ch}", rgb_video=rgb_video, seed=args.seed, num_frames=F)
            save_video(vid, str(p), fps=args.fps, quality=args.quality)
            print(f" Saved {ch} to {p}")
        pred_paths[ch] = p

    # -------- 2) Intrinsic Video Sampling (keyframe=edit_type, rest=random non-edit types) --------
    paths = {t: pred_paths[t] for t in TYPE_ORDER}
    intrinsic_video, frame_type_ids = intrinsic_video_sample(paths=paths, F=F, H=H, W=W, edit_type=args.edit_type, edit_x_path=args.edit_x_path, drop_type=args.drop_type)

    if args.save_mixed:
        mixed_path = out_dir / f"{base}_{args.task}_intrinsic_video.mp4"
        iio.mimwrite(mixed_path, [f.astype(np.uint8) for f in intrinsic_video], fps=args.fps)
        print(f"🧩 Saved intrinsic video: {mixed_path}")

    # -------- 3) X + ref -> RGB (forward rendering) --------
    pipe_x2rgb = build_pipe(args.device, args.torch_dtype, change_dim=args.x2rgb_change_dim, num_types=len(TYPE_ORDER))
    pipe_x2rgb.dit.load_state_dict(load_file(args.x2rgb_ckpt, device="cpu"))
    print(f"✅ [X2RGB] Loaded forward renderer weights from: {args.x2rgb_ckpt}")
    pipe_x2rgb.enable_vram_management()

    ref_img = Image.open(args.ref_rgb_path).convert("RGB").resize((W, H))

    out = pipe_x2rgb(
        prompt="#rgb",
        rgbx_video=intrinsic_video, 
        input_image=ref_img,
        frame_type_ids=frame_type_ids,
        seed=args.seed,
        num_frames=F,
        rgb_dropout=args.rgb_dropout,
        ref_cfg_scale=args.ref_cfg_scale,
    )

    out_path = out_dir / f"{base}_{args.task}_edit.mp4"
    save_video(out, str(out_path), fps=args.fps, quality=args.quality)
    print(f"✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
