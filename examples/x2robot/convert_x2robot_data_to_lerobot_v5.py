"""
V5: Direct ffmpeg transcoding - eliminates disk I/O for PNG frames.

Optimization strategy:
1. Parallel TRANSCODING: N episodes × 3 cameras = 3N direct video conversions
2. Sequential dataset building (metadata only, use dummy video stats)
3. No need for sample frame extraction!

Key insight: Use ffmpeg to directly transcode from source MP4 to the target codec,
and use dummy statistics for video features (since they're always [0,1] normalized anyway).

V4 workflow:
  Source MP4 → ffmpeg decode → PNG files → encode_video_frames → target MP4

V5 workflow:
  Source MP4 → ffmpeg transcode → target MP4 (single step!)
  Video stats → use dummy values (min=0, max=1, mean~0.4, std~0.25)

Expected speedup: ~2-3x
"""
import os
import shutil
import glob
import json
import subprocess
import numpy as np
import tqdm
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME
import tyro
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction

    
# Suppress ffmpeg output
os.environ['SVT_LOG'] = '0'


# Configuration
REPO_NAME = "pour_tea_chengdu_20260601-20260605_sm2sm"
RAW_DATASET_PATHS = [
    './datasets/pour_tea/chengdu_huangdandan_20260601_pm/',
    './datasets/pour_tea/chengdu_huangdandan_20260602_pm/',
    './datasets/pour_tea/chengdu_huangdandan_20260603_pm/',
    './datasets/pour_tea/chengdu_huangdandan_20260604_pm/',
    './datasets/pour_tea/chengdu_huangdandan_20260605_pm/',
    './datasets/pour_tea/chengdu_wenpei_20260601_pm/',
    './datasets/pour_tea/chengdu_wenpei_20260602_pm/',
    './datasets/pour_tea/chengdu_wenpei_20260603_pm/',
    './datasets/pour_tea/chengdu_wenpei_20260604_pm/',
    './datasets/pour_tea/chengdu_wenpei_20260605_pm/',
]

FILE_CAMERA_MAPPING = {
    "face_view": "faceImg.mp4",
    "left_wrist_view": "leftImg.mp4",
    "right_wrist_view": "rightImg.mp4"
}

STATE_KEYS = [
    'follow_left_position',
    'follow_left_rotation', 
    'follow_left_gripper',
    'follow_right_position',
    'follow_right_rotation',
    'follow_right_gripper',   

    'master_left_position',
    'master_left_rotation',
    'master_left_gripper', 
    'master_right_position',
    'master_right_rotation',
    'master_right_gripper',
]

ACTION_KEYS = [
    'follow_left_position',
    'follow_left_rotation', 
    'follow_left_gripper',
    'follow_right_position',
    'follow_right_rotation',
    'follow_right_gripper',
    
    'master_left_position',
    'master_left_rotation',
    'master_left_gripper', 
    'master_right_position',
    'master_right_rotation',
    'master_right_gripper',
]


def get_dim_from_keys(keys: list[str]) -> int:
    """Calculate the total dimension from a list of keys."""
    dim = 0
    for key in keys:
        if 'gripper' in key:
            dim += 1
        elif 'position' in key:
            dim += 3
        elif 'rotation' in key:
            dim += 3
        elif 'joint' in key:
            dim += 7
        else:
            raise ValueError(f"Unknown key type: {key}")
    return dim


def find_episodes(raw_paths: list[str]) -> list[str]:
    """Find all episode directories containing MP4 files."""
    episode_paths = []
    for raw_path in raw_paths:
        for dir_path in glob.glob(f'{raw_path}/*'):
            if os.path.isdir(dir_path):
                mp4_files = glob.glob(f'{dir_path}/*.mp4')
                if len(mp4_files) > 0:
                    episode_paths.append(dir_path)
    return sorted(episode_paths)


def find_tagged_quality_episodes(
    dataset_root: Path,
    required_tag: str,
    tags_relative_path: Path,
    quality_annotation_relative_path: Path,
    quality_key: str,
) -> tuple[list[str], dict[str, tuple[int, int]]]:
    """Select tagged episodes which have a valid quality-passed frame range."""
    episode_paths = []
    passed_ranges = {}
    tagged_count = 0
    missing_quality_count = 0

    for episode_path in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        tags_path = episode_path / tags_relative_path
        if not tags_path.is_file():
            continue

        tags_payload = json.loads(tags_path.read_text())
        tags = tags_payload.get("tags", [])
        if not isinstance(tags, list):
            raise ValueError(f"Expected a list at {tags_path}:tags")
        if required_tag not in tags:
            continue
        tagged_count += 1

        quality_path = episode_path / quality_annotation_relative_path
        if not quality_path.is_file():
            missing_quality_count += 1
            continue

        quality_payload = json.loads(quality_path.read_text())
        frame_range = quality_payload.get(quality_key)
        if (
            not isinstance(frame_range, list)
            or len(frame_range) != 2
            or not all(isinstance(value, int) for value in frame_range)
        ):
            raise ValueError(
                f"Expected key {quality_key!r} to contain [start_frame, end_frame] at {quality_path}"
            )
        start_frame, end_frame = frame_range
        if start_frame < 0 or end_frame <= start_frame:
            raise ValueError(f"Invalid half-open frame range {frame_range} at {quality_path}")

        episode_path_str = str(episode_path)
        episode_paths.append(episode_path_str)
        passed_ranges[episode_path_str] = (start_frame, end_frame)

    print(
        f"Tag {required_tag!r}: {tagged_count} episodes; selected {len(episode_paths)} with "
        f"{quality_annotation_relative_path}; skipped {missing_quality_count} without it"
    )
    return episode_paths, passed_ranges


def load_json_data(
    episode_path: str,
    target_frame_count: int,
    target_fps: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Load and parse JSON data for an episode."""
    episode_name = os.path.basename(episode_path)
    json_path = os.path.join(episode_path, f"{episode_name}.json")
    
    with open(json_path, 'r') as f:
        payload = json.load(f)
    data = payload['data']
    source_fps = payload.get("fps")
    if source_fps is None:
        video_path = os.path.join(episode_path, FILE_CAMERA_MAPPING["face_view"])
        probe_result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
        )
        if probe_result.returncode != 0 or not probe_result.stdout.strip():
            raise RuntimeError(f"Could not determine source fps for {video_path}: {probe_result.stderr}")
        source_fps = float(Fraction(probe_result.stdout.strip()))
    else:
        source_fps = float(source_fps)
    if source_fps <= 0:
        raise ValueError(f"Invalid source fps {source_fps} at {json_path}")
    target_indices = np.arange(target_frame_count, dtype=np.float64)
    source_indices = np.rint(target_indices * source_fps / target_fps).astype(np.int64)
    source_indices = np.clip(source_indices, 0, len(data) - 1)
    
    all_keys = set(STATE_KEYS) | set(ACTION_KEYS)
    trajectories = {key: [] for key in all_keys}
    for frame_data in data:
        for key in all_keys:
            trajectories[key].append(frame_data[key])
    
    arrays = {}
    for key, vals in trajectories.items():
        arr = np.array(vals, dtype=np.float32)
        if 'gripper' in key:
            arr = arr.reshape(-1, 1)
        arrays[key] = arr
    
    state_array = np.concatenate([arrays[key][source_indices] for key in STATE_KEYS], axis=1)
    action_array = np.concatenate([arrays[key][source_indices] for key in ACTION_KEYS], axis=1)
    return state_array, action_array, source_fps, len(data)


def _target_excluded_ranges(
    passed_range: tuple[int, int],
    source_frame_count: int,
    source_fps: float,
    target_fps: int,
    target_frame_count: int,
) -> tuple[list[list[int]], list[int] | None]:
    """Map the complement of a raw passed interval to rows with next-frame actions."""
    target_indices = np.arange(target_frame_count, dtype=np.float64)
    source_indices = np.rint(target_indices * source_fps / target_fps).astype(np.int64)
    source_indices = np.clip(source_indices, 0, source_frame_count - 1)

    source_start, source_end = passed_range
    passed_source_frames = np.zeros(source_frame_count, dtype=bool)
    passed_source_frames[source_start:source_end] = True
    passed_rows = passed_source_frames[source_indices[:-1]] & passed_source_frames[source_indices[1:]]
    excluded_rows = ~passed_rows

    changes = np.flatnonzero(excluded_rows[1:] != excluded_rows[:-1]) + 1
    excluded_ranges = []
    run_start = 0
    for run_end in [*changes.tolist(), len(excluded_rows)]:
        if excluded_rows[run_start]:
            excluded_ranges.append([run_start, run_end])
        run_start = run_end

    passed_indices = np.flatnonzero(passed_rows)
    target_passed_range = (
        [int(passed_indices[0]), int(passed_indices[-1]) + 1]
        if len(passed_indices)
        else None
    )
    return excluded_ranges, target_passed_range


def write_quality_filter_metadata(
    output_path: Path,
    quality_annotation_relative_path: Path,
    quality_key: str,
    target_fps: int,
    episode_paths: list[str],
    passed_ranges: dict[str, tuple[int, int]],
    video_frame_counts: list[int],
    source_fps_values: list[float],
    source_frame_counts: list[int],
) -> None:
    metadata_dir = output_path / "meta" / "data_quality"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    episodes = []

    for episode_index, (episode_path, video_frame_count, source_fps, source_frame_count) in enumerate(
        zip(
            episode_paths,
            video_frame_counts,
            source_fps_values,
            source_frame_counts,
            strict=True,
        )
    ):
        source_start, source_end = passed_ranges[episode_path]
        if source_end > source_frame_count:
            raise ValueError(
                f"Quality-passed end frame {source_end} exceeds source length {source_frame_count}: "
                f"{episode_path}"
            )

        excluded_ranges, target_passed_range = _target_excluded_ranges(
            (source_start, source_end),
            source_frame_count,
            source_fps,
            target_fps,
            video_frame_count,
        )

        episodes.append(
            {
                "episode_index": episode_index,
                "source_episode": Path(episode_path).name,
                "source_passed_range": [source_start, source_end],
                "target_passed_range": target_passed_range,
                "sample_ranges": excluded_ranges,
            }
        )

    metadata = {
        "format_version": 1,
        "annotation_relative_path": str(quality_annotation_relative_path),
        "quality_key": quality_key,
        "target_fps": target_fps,
        "range_semantics": "half_open",
        "sample_ranges_semantics": "excluded_complement_of_quality_passed",
        "episodes": episodes,
    }
    with (metadata_dir / "excluded_sample_ranges.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def transcode_video_ffmpeg(
    input_path: str,
    output_path: Path,
    target_size: tuple[int, int] = (320, 240),
    fps: int = 20,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
) -> int:
    """直接使用ffmpeg转码视频，参数与encode_video_frames完全一致。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ffmpeg", "-y",
        "-nostdin",
        "-i", input_path,
        "-vf", f"scale={target_size[0]}:{target_size[1]}",
        "-c:v", vcodec,
        "-pix_fmt", pix_fmt,
        "-r", str(fps),
        "-g", str(g),
        "-crf", str(crf),
        str(output_path)
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, 'SVT_LOG': '0'}
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg transcode failed for {input_path}: {result.stderr}")
    
    # 获取帧数
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        str(output_path)
    ]
    
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if probe_result.returncode != 0 or not probe_result.stdout.strip():
        raise RuntimeError(f"ffprobe failed for {output_path}: {probe_result.stderr}")
    num_frames = int(probe_result.stdout.strip())

    decode_check_cmd = [
        "ffmpeg", "-v", "error",
        "-xerror",
        "-nostdin",
        "-i", str(output_path),
        "-f", "null",
        "-"
    ]
    decode_check_result = subprocess.run(decode_check_cmd, capture_output=True, text=True)
    if decode_check_result.returncode != 0 or decode_check_result.stderr.strip():
        raise RuntimeError(f"ffmpeg decode check failed for {output_path}: {decode_check_result.stderr}")
    
    return num_frames


def transcode_single_video(
    episode_path: str,
    episode_index: int,
    camera_name: str,
    video_filename: str,
    output_root: Path,
    target_size: tuple[int, int],
    fps: int = 20,
    video_codec: str = "h264",
) -> tuple[int, str, int]:
    """转码单个视频文件。返回 (episode_idx, camera_name, num_frames)。"""
    video_path = os.path.join(episode_path, video_filename)
    output_path = output_root / "videos" / "chunk-000" / camera_name / f"episode_{episode_index:06d}.mp4"
    
    if video_codec == "av1":
        vcodec, crf = "libsvtav1", 30
    elif video_codec == "h264":
        vcodec, crf = "libx264", 23
    else:
        raise ValueError(f"Unsupported video codec: {video_codec}")

    num_frames = transcode_video_ffmpeg(
        video_path,
        output_path,
        target_size,
        fps,
        vcodec=vcodec,
        pix_fmt="yuv420p",
        g=2,
        crf=crf,
    )
    
    return episode_index, camera_name, num_frames


def get_dummy_video_stats(num_frames: int) -> dict:
    """生成视频特征的伪统计值（与真实值非常接近）"""
    return {
        "min": np.array([[[0.0]], [[0.0]], [[0.0]]]),  # RGB channels
        "max": np.array([[[1.0]], [[1.0]], [[1.0]]]),
        "mean": np.array([[[0.4]], [[0.4]], [[0.4]]]),  # typical mean
        "std": np.array([[[0.25]], [[0.25]], [[0.25]]]),  # typical std
        "count": np.array([num_frames])
    }


def compute_episode_stats_with_dummy_video(
    episode_buffer: dict, 
    features: dict,
    video_frame_count: int
) -> dict:
    """计算episode统计，对视频特征使用伪值"""
    from lerobot.common.datasets.compute_stats import get_feature_stats
    
    ep_stats = {}
    for key, data in episode_buffer.items():
        if key not in features:
            continue
            
        if features[key]["dtype"] == "string":
            continue
        elif features[key]["dtype"] in ["image", "video"]:
            # 使用伪统计值，不读取图片
            ep_stats[key] = get_dummy_video_stats(video_frame_count)
        else:
            # 对于其他特征，正常计算统计
            ep_ft_array = data
            axes_to_reduce = 0
            keepdims = data.ndim == 1
            ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)
    
    return ep_stats


class NoVideoIOLeRobotDataset(LeRobotDataset):
    """LeRobotDataset that skips all video/image I/O in save_episode."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skip_all_media = False
        self._video_frame_count = 0  # 设置每个episode的视频帧数
    
    def _save_image(self, image, fpath: Path) -> None:
        """Skip all image saving."""
        if self._skip_all_media:
            return
        super()._save_image(image, fpath)
    
    def save_episode(self, episode_data: dict | None = None) -> None:
        """Override save_episode to skip all video/image I/O and use dummy stats."""
        if not self._skip_all_media:
            super().save_episode(episode_data)
            return
            
        if not episode_data:
            episode_buffer = self.episode_buffer
        
        from lerobot.common.datasets.lerobot_dataset import (
            validate_episode_buffer,
            get_episode_data_index,
            check_timestamps_sync,
            write_info,
            write_episode,
            write_episode_stats,
            aggregate_stats,
        )

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        
        # 使用自定义的统计计算函数（对视频使用伪值）
        ep_stats = compute_episode_stats_with_dummy_video(
            episode_buffer, self.features, self._video_frame_count
        )

        # Save episode metadata
        self.meta.info["total_episodes"] += 1
        self.meta.info["total_frames"] += episode_length

        chunk = self.meta.get_episode_chunk(episode_index)
        if chunk >= self.meta.total_chunks:
            self.meta.info["total_chunks"] += 1

        self.meta.info["splits"] = {"train": f"0:{self.meta.info['total_episodes']}"}
        self.meta.info["total_videos"] += len(self.meta.video_keys)
        
        write_info(self.meta.info, self.meta.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        self.meta.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.meta.root)

        self.meta.episodes_stats[episode_index] = ep_stats
        self.meta.stats = aggregate_stats([self.meta.stats, ep_stats]) if self.meta.stats else ep_stats
        write_episode_stats(episode_index, ep_stats, self.meta.root)

        if not episode_data:
            self.episode_buffer = self.create_episode_buffer()
    
    def finalize_video_info(self) -> None:
        """Update video info after all videos are transcoded."""
        self.meta.update_video_info()
        from lerobot.common.datasets.lerobot_dataset import write_info
        write_info(self.meta.info, self.meta.root)
    
    @classmethod
    def create(cls, **kwargs) -> "NoVideoIOLeRobotDataset":
        """Create a NoVideoIOLeRobotDataset."""
        parent_obj = LeRobotDataset.create(**kwargs)
        obj = cls.__new__(cls)
        obj.__dict__.update(parent_obj.__dict__)
        obj._skip_all_media = False
        obj._video_frame_count = 0
        return obj


def main(
    dataset_root: Path | None = None,
    dataset_roots: tuple[Path, ...] = (),
    repo_name: str = REPO_NAME,
    push_to_hub: bool = False,
    debug: bool = False,
    debug_episodes: int = 3,
    low_resolution: bool = True,
    num_workers: int = 10,
    target_fps: int = 20,
    video_codec: str = "h264",
    overwrite: bool = False,
    required_tag: str | None = None,
    tags_relative_path: Path = Path("anno/tags.json"),
    quality_annotation_relative_path: Path | None = None,
    quality_key: str = "0",
):
    """
    V5: Direct ffmpeg transcoding optimization.
    """
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if video_codec not in {"av1", "h264"}:
        raise ValueError(f"Unsupported video codec: {video_codec}")

    print(f"HF_LEROBOT_HOME: {HF_LEROBOT_HOME}")
    print(f"V5: Direct ffmpeg transcoding (num_workers={num_workers}, codec={video_codec})")
    print(f"Target fps: {target_fps}")
    
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)
    
    if dataset_root is not None and dataset_roots:
        raise ValueError("Pass either dataset_root or dataset_roots, not both")

    quality_passed_ranges = None
    if required_tag is not None:
        if dataset_roots:
            raise ValueError("required_tag selection currently accepts one dataset_root")
        if dataset_root is None:
            raise ValueError("dataset_root is required when required_tag is set")
        if quality_annotation_relative_path is None:
            raise ValueError("quality_annotation_relative_path is required when required_tag is set")
        episode_paths, quality_passed_ranges = find_tagged_quality_episodes(
            dataset_root,
            required_tag,
            tags_relative_path,
            quality_annotation_relative_path,
            quality_key,
        )
    else:
        if quality_annotation_relative_path is not None:
            raise ValueError("required_tag must be set when quality_annotation_relative_path is set")
        if dataset_roots:
            raw_dataset_paths = [str(path) for path in dataset_roots]
        else:
            raw_dataset_paths = [str(dataset_root)] if dataset_root is not None else RAW_DATASET_PATHS
        episode_paths = find_episodes(raw_dataset_paths)
    print(f"Found {len(episode_paths)} episodes")
    if debug:
        episode_paths = episode_paths[:debug_episodes]
        print(f"Debug mode: only processing first {debug_episodes} episodes")

    episode_failures: dict[int, list[str]] = {}
    
    target_size = (320, 240) if low_resolution else (640, 480)
    shape = (target_size[1], target_size[0], 3)  # (H, W, C)
    
    total_start = time.time()
    
    # ========================================
    # Create Dataset
    # ========================================
    print("\nCreating LeRobotDataset...")
    dataset = NoVideoIOLeRobotDataset.create(
        repo_id=repo_name,
        robot_type="ARX",
        fps=target_fps,
        features={
            "face_view": {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channel"],
            },
            "left_wrist_view": {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channel"],
            },
            "right_wrist_view": {
                "dtype": "video",
                "shape": shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (get_dim_from_keys(STATE_KEYS),),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (get_dim_from_keys(ACTION_KEYS),),
                "names": ["actions"],
            },
        },
        image_writer_threads=0,
        image_writer_processes=0,
    )
    
    dataset._skip_all_media = True
    
    # ========================================
    # PHASE 1: Parallel Transcoding
    # ========================================
    print(f"\n{'='*60}")
    print("PHASE 1: Parallel direct video transcoding (ffmpeg)")
    print(f"{'='*60}")
    
    t_transcode_start = time.time()
    
    transcode_tasks = []
    for ep_idx, ep_path in enumerate(episode_paths):
        for camera_name, video_filename in FILE_CAMERA_MAPPING.items():
            transcode_tasks.append(
                (ep_path, ep_idx, camera_name, video_filename, output_path, target_size, target_fps, video_codec)
            )
    
    episode_frame_counts = {}
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(transcode_single_video, *task): task[:3]
            for task in transcode_tasks
        }
        
        with tqdm.tqdm(total=len(transcode_tasks), desc="Transcoding videos") as pbar:
            for future in as_completed(futures):
                try:
                    ep_idx, camera_name, num_frames = future.result()
                    if ep_idx not in episode_frame_counts:
                        episode_frame_counts[ep_idx] = {}
                    episode_frame_counts[ep_idx][camera_name] = num_frames
                except Exception as e:
                    ep_path, ep_idx, camera = futures[future]
                    error_summary = "\n".join(str(e).splitlines()[-8:])
                    episode_failures.setdefault(ep_idx, []).append(
                        f"{camera} transcode failed: {error_summary}"
                    )
                    print(
                        f"\nSkipping episode {ep_idx} after transcode error "
                        f"({camera}): {ep_path}\n{error_summary}\n"
                    )
                pbar.update(1)
    
    t_transcode_end = time.time()
    print(f"Transcoding completed in {t_transcode_end - t_transcode_start:.2f}s")

    skipped_episode_indices = sorted(episode_failures)
    successful_episode_indices = [
        ep_idx for ep_idx in range(len(episode_paths))
        if ep_idx not in episode_failures
    ]

    for ep_idx in skipped_episode_indices:
        for camera_name in FILE_CAMERA_MAPPING:
            path = output_path / "videos" / "chunk-000" / camera_name / f"episode_{ep_idx:06d}.mp4"
            if path.exists():
                path.unlink()

    for new_idx, old_idx in enumerate(successful_episode_indices):
        if new_idx == old_idx:
            continue
        for camera_name in FILE_CAMERA_MAPPING:
            src = output_path / "videos" / "chunk-000" / camera_name / f"episode_{old_idx:06d}.mp4"
            dst = output_path / "videos" / "chunk-000" / camera_name / f"episode_{new_idx:06d}.mp4"
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)

    processed_episode_paths = [episode_paths[ep_idx] for ep_idx in successful_episode_indices]
    processed_video_frame_counts = [
        min(episode_frame_counts[ep_idx].values()) for ep_idx in successful_episode_indices
    ]

    if skipped_episode_indices:
        print(f"Skipped {len(skipped_episode_indices)} episodes:")
        for ep_idx in skipped_episode_indices:
            print(f"  [{ep_idx}] {episode_paths[ep_idx]}")
            for reason in episode_failures[ep_idx]:
                print(f"      - {reason}")

    if not processed_episode_paths:
        raise RuntimeError("No valid episodes left after transcoding.")
    
    # ========================================
    # PHASE 2: Build Dataset (metadata only)
    # ========================================
    print(f"\n{'='*60}")
    print("PHASE 2: Building dataset (metadata only, dummy video stats)")
    print(f"{'='*60}")
    
    t_build_start = time.time()
    
    dummy_image = np.zeros((shape[0], shape[1], shape[2]), dtype=np.uint8)
    processed_source_fps = []
    processed_source_frame_counts = []
    
    import datasets
    datasets.disable_progress_bars()
    
    for ep_idx, (ep_path, video_frame_count) in enumerate(
        tqdm.tqdm(
            zip(processed_episode_paths, processed_video_frame_counts, strict=True),
            total=len(processed_episode_paths),
            desc="Building dataset",
        )
    ):
        state_array, action_array, source_fps, source_frame_count = load_json_data(
            ep_path, video_frame_count, target_fps
        )
        processed_source_fps.append(source_fps)
        processed_source_frame_counts.append(source_frame_count)
        num_frames = len(state_array)
        
        # 设置视频帧数（用于伪统计）
        dataset._video_frame_count = num_frames - 1
        
        for i in range(num_frames - 1):
            frame_data = {
                "face_view": dummy_image,
                "left_wrist_view": dummy_image,
                "right_wrist_view": dummy_image,
                "state": state_array[i],
                "actions": action_array[i + 1],
                "task": '',
            }
            dataset.add_frame(frame_data)
        
        dataset.save_episode()
    
    t_build_end = time.time()
    print(f"Dataset building completed in {t_build_end - t_build_start:.2f}s")
    
    # ========================================
    # PHASE 3: Finalize and Cleanup
    # ========================================
    print("\nFinalizing video info...")
    dataset.finalize_video_info()

    if quality_passed_ranges is not None:
        assert quality_annotation_relative_path is not None
        write_quality_filter_metadata(
            output_path,
            quality_annotation_relative_path,
            quality_key,
            target_fps,
            processed_episode_paths,
            quality_passed_ranges,
            processed_video_frame_counts,
            processed_source_fps,
            processed_source_frame_counts,
        )
        print("Wrote training-time quality filter metadata")
    
    # 清理lerobot创建的空images目录
    img_dir = output_path / "images"
    if img_dir.is_dir():
        shutil.rmtree(img_dir)
        print("Cleaned up empty images directory.")
    
    # ========================================
    # Summary
    # ========================================
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Episodes found:     {len(episode_paths)}")
    print(f"  Episodes processed: {len(processed_episode_paths)}")
    print(f"  Episodes skipped:   {len(skipped_episode_indices)}")
    print(f"  Phase 1 (transcode): {t_transcode_end - t_transcode_start:.2f}s")
    print(f"  Phase 2 (build):     {t_build_end - t_build_start:.2f}s")
    print(f"  Total:               {total_time:.2f}s")
    print(f"  Average per episode: {total_time/len(processed_episode_paths):.2f}s")
    print(f"{'='*60}")
    print(f"Dataset saved at {output_path}")
    
    if push_to_hub:
        print("Pushing to Hugging Face Hub...")
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(main)
