#!/usr/bin/env python3
"""
将自适应加速因子曲线绘制为图，并实时叠加到 3 个 MP4 视频的左上角。

示例:
  python visualize_factors.py \
    --data_dir /mnt/public3/datasets/x1pro/table_clean_sop_0720_v2 \
    --chunk_size 30 \
    --ratio 0.4 \
    --output_dir ./output \
    --episode 0
"""

import argparse
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils import load_factors_json


def make_factor_overlay(factors, current_frame, plot_size, dpi=100):
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(plot_size / dpi, plot_size / dpi), dpi=dpi)
    fig.patch.set_facecolor((0.06, 0.06, 0.06, 0.85))

    n = len(factors)
    frames = np.arange(n)
    ax.plot(frames, factors, color="#00d4aa", linewidth=1.2, alpha=0.9)

    x_min = max(0, current_frame - int(n * 0.1))
    x_max = min(n, current_frame + max(int(n * 0.1), 1))
    ax.set_xlim(x_min, x_max)

    y_min = max(0.5, np.min(factors) - 0.3)
    y_max = min(6.0, np.max(factors) + 0.3)
    ax.set_ylim(y_min, y_max)

    if 0 <= current_frame < n:
        cur_val = factors[current_frame]
        ax.axvline(x=current_frame, color="#ff5555", linewidth=1.5, alpha=0.8)
        ax.scatter([current_frame], [cur_val], color="#ff5555", s=40, zorder=5)

        ax.set_title(
            f"Frame: {current_frame:>5d}  |  Factor: {cur_val:.2f}",
            fontsize=9,
            color="#ff5555",
            pad=4,
            fontfamily="monospace",
        )
    else:
        ax.set_title("Optimal Speedup Factor", fontsize=9, pad=4, fontfamily="monospace")

    ax.set_xlabel("Frame", fontsize=7, color="#aaaaaa")
    ax.set_ylabel("Factor", fontsize=7, color="#aaaaaa")
    ax.tick_params(labelsize=6, colors="#999999")
    ax.grid(True, alpha=0.15, linewidth=0.5)

    fig.tight_layout(pad=0.5)

    # Render figure to RGBA numpy array using a non-interactive approach
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return buf


def overlay_plot_on_frame(frame, overlay_rgba, margin_x=10, margin_y=10):
    h, w = frame.shape[:2]
    oh, ow = overlay_rgba.shape[:2]

    oh = min(oh, h - 2 * margin_y)
    ow = min(ow, w - 2 * margin_x)
    if oh != overlay_rgba.shape[0] or ow != overlay_rgba.shape[1]:
        overlay_rgba = cv2.resize(overlay_rgba, (ow, oh), interpolation=cv2.INTER_AREA)

    alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    overlay_bgr = cv2.cvtColor(overlay_rgba[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)

    roi = frame[margin_y:margin_y + oh, margin_x:margin_x + ow].astype(np.float32)
    blended = overlay_bgr * alpha + roi * (1.0 - alpha)
    frame[margin_y:margin_y + oh, margin_x:margin_x + ow] = blended.astype(np.uint8)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Visualize adaptive speedup factors overlaid on MP4 videos")
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/public3/datasets/x1pro/table_clean_sop_0720_v2",
                        help="Dataset root directory")
    parser.add_argument("--chunk_size", type=int, required=True,
                        help="Action chunk size (e.g. 30)")
    parser.add_argument("--ratio", type=float, required=True,
                        help="Ratio for dynamic threshold (e.g. 0.4)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for processed videos")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index (0-based, sorted order)")
    parser.add_argument("--plot_size", type=int, default=400,
                        help="Plot overlay size in pixels (square)")

    args = parser.parse_args()

    episode_dirs = sorted([
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d)) and not d.startswith(".")
    ])

    if args.episode < 0 or args.episode >= len(episode_dirs):
        print(f"Episode index {args.episode} out of range (0-{len(episode_dirs) - 1})")
        return

    ep_name = episode_dirs[args.episode]
    ep_dir = os.path.join(args.data_dir, ep_name)

    factors = load_factors_json(ep_dir, args.chunk_size, args.ratio)
    if factors is None:
        ratio_str = str(args.ratio).replace(".", "_")
        dir_name = f"c{args.chunk_size}_{ratio_str}"
        print(f"Factor file not found: {ep_dir}/factor/{dir_name}/adaptive_factor.json")
        return

    factors = np.asarray(factors)
    n_frames = len(factors)
    print(f"Episode: {ep_name}")
    print(f"Total frames: {n_frames}")
    print(f"Factor range: [{factors.min():.2f}, {factors.max():.2f}]")
    print(f"Mean factor: {factors.mean():.2f}")

    os.makedirs(args.output_dir, exist_ok=True)

    cameras = {
        "face": "faceImg.mp4",
        "left_wrist": "leftImg.mp4",
        "right_wrist": "rightImg.mp4",
    }

    for cam_name, cam_file in cameras.items():
        cam_path = os.path.join(ep_dir, cam_file)
        if not os.path.exists(cam_path):
            print(f"  Skipping {cam_name}: video not found")
            continue

        cap = cv2.VideoCapture(cam_path)
        if not cap.isOpened():
            print(f"  Skipping {cam_name}: cannot open video")
            continue

        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = os.path.join(args.output_dir, f"{ep_name}_{cam_name}_overlay.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

        print(f"\n  [{cam_name}] {cam_path}")
        print(f"    Size: {out_w}x{out_h}, FPS: {fps:.1f}, Frames: {total_frames}")
        print(f"    Output: {out_path}")

        min_len = min(total_frames, n_frames)

        for i in range(min_len):
            ret, frame = cap.read()
            if not ret:
                break

            overlay = make_factor_overlay(factors, i, args.plot_size)
            frame = overlay_plot_on_frame(frame, overlay)

            writer.write(frame)

            if (i + 1) % 500 == 0 or i == min_len - 1:
                print(f"    ... {i + 1}/{min_len} frames processed")

        writer.release()
        cap.release()

    print("\nDone.")


if __name__ == "__main__":
    main()
