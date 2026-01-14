import os
import argparse
import torch

from vrgbx import save_video, VideoData, load_state_dict
from vrgbx.pipelines.vrgbx_model import VRGBXPipeline, ModelConfig


def load_video_slice(video_path: str, height: int, width: int, start: int, num_frames: int):
    video = VideoData(video_path, height=height, width=width)
    end = start + num_frames
    return [video[i] for i in range(start, end)], start, end


def parse_args():
    p = argparse.ArgumentParser("RGB->X video inference (local ckpt)")

    # input / output
    p.add_argument("--video_path", type=str, default="examples/input_videos/Evermotion_CreativeLoft.mp4")
    p.add_argument("--ckpt_path", type=str, default="models/V-RGBX/vrgbx_inverse_renderer.safetensors", help="Local .safetensors path")
    p.add_argument("--save_dir", type=str, default="output/inverse_rendering")

    # video slicing
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--num_frames", type=int, default=49)

    # generation
    p.add_argument("--channels", nargs="+", default=["albedo", "normal", "material", "irradiance"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tiled", action="store_true")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--quality", type=int, default=5)

    # pipeline
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--torch_dtype", type=str, default="bfloat16")

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
        change_dim=32,
    )
    return pipe

def main():
    args = parse_args()

    out_dir = args.save_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Output] {out_dir}")

    print("[1/4] Build pipeline...")
    pipe = build_pipe(args.device, args.torch_dtype)

    print("[2/4] Load checkpoint...")
    state_dict = load_state_dict(args.ckpt_path)
    pipe.dit.load_state_dict(state_dict)
    print(f"✅ [RGB2X] Loaded inverse renderer weights from: {args.ckpt_path}")
    pipe.enable_vram_management()

    print("[3/4] Load video frames...")
    rgb_video, sid, eid = load_video_slice(
        args.video_path, args.height, args.width, args.start, args.num_frames
    )
    print(f"Loaded frames [{sid}, {eid}) from {args.video_path}")

    channels = args.channels
    print(f"[4/4] Run channels: {channels}")
    base = os.path.basename(args.video_path).replace('.mp4', '')

    for ch in channels:
        video = pipe(
            prompt=f"#{ch}",
            rgb_video=rgb_video,
            seed=args.seed,
            tiled=args.tiled,
            num_frames=args.num_frames,
        )
        save_path = os.path.join(out_dir, f"{base}_video_rgb2x_{sid}_{eid}_{ch}.mp4")
        save_video(video, save_path, fps=args.fps, quality=args.quality)
        print(f"✅ Saved: {save_path}")


if __name__ == "__main__":
    main()
