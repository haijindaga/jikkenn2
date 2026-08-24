#!/usr/bin/env python3
"""Find the tool in the captured image with SAM3, and cut out its points.

Phase 2 replaces one input: where the tool is stops coming from the simulator
and starts coming from the camera.  Which part of it is dangerous is still
ground truth; SAM3 takes that over in Phase 3.

    conda activate env_isaaclab
    python scripts/segment_tool.py --capture outputs/trial_001 --prompt "red block"

Isaac is not started: this reads the frame the capture already saved, so only
SAM3 is on the GPU.  The mask, the overlay and the masked point cloud are all
written out, because a segmentation nobody looked at is a segmentation nobody
checked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from jikkenn2 import ground_truth as gt  # noqa: E402
from jikkenn2.pointcloud import write_colored_cloud  # noqa: E402
from jikkenn2.scene_spec import DEFAULT_SCENE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--prompt",
        default=None,
        help="Short noun phrase for SAM3; defaults to the grasped part's own prompt",
    )
    parser.add_argument("--model-id", default="facebook/sam3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face network access instead of requiring the cache",
    )
    return parser.parse_args()


def run_sam3(rgb: np.ndarray, prompt: str, args) -> dict:
    """The Hugging Face text-prompted recipe, unchanged."""
    import torch
    from PIL import Image
    from transformers import Sam3Model, Sam3Processor

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    local_only = not args.allow_download
    processor = Sam3Processor.from_pretrained(args.model_id, local_files_only=local_only)
    model = Sam3Model.from_pretrained(args.model_id, local_files_only=local_only)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    model = model.to(device=args.device, dtype=dtype).eval()

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_instance_segmentation(
        outputs,
        threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    return {
        "masks": result["masks"].detach().cpu().numpy().astype(bool),
        "boxes": result["boxes"].detach().cpu().numpy().astype(np.float32),
        "scores": result["scores"].detach().cpu().numpy().astype(np.float32),
    }


def save_overlay(path: Path, rgb: np.ndarray, mask: np.ndarray) -> None:
    from PIL import Image

    overlay = np.asarray(rgb, dtype=np.float32).copy()
    overlay[mask] = overlay[mask] * 0.45 + np.array([0.0, 255.0, 80.0]) * 0.55
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(path)


def agreement_with_ground_truth(
    mask: np.ndarray, truth: np.ndarray
) -> dict:
    """How the mask compares with where the tool really is.

    Reported, never used to correct the mask: this is the number that says what
    perception cost, and it would be worthless if it fed back into the result.
    """
    intersection = int(np.count_nonzero(mask & truth))
    union = int(np.count_nonzero(mask | truth))
    return {
        "usage": "diagnostic only; never used to modify the mask",
        "intersection_over_union": round(intersection / union, 4) if union else None,
        "mask_pixels": int(np.count_nonzero(mask)),
        "true_pixels": int(np.count_nonzero(truth)),
        "missed_pixels": int(np.count_nonzero(truth & ~mask)),
        "spilled_pixels": int(np.count_nonzero(mask & ~truth)),
    }


def main() -> int:
    args = parse_args()
    scene = DEFAULT_SCENE
    output = args.output or (args.capture / "segmentation")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "segmentation_check.json"

    try:
        rgb = np.load(args.capture / "rgb.npy")
        depth = np.load(args.capture / "depth_m.npy")
        points_camera = np.load(args.capture / "points_camera.npy")
        points_world = np.load(args.capture / "points_world.npy")
        tool_pose = np.load(args.capture / "tool_pose_world.npy")

        prompt = args.prompt or scene.part(scene.grasp_part_name).prompt
        if not prompt:
            raise ValueError(
                f"no prompt for part {scene.grasp_part_name!r}; pass --prompt"
            )
        prediction = run_sam3(rgb, prompt, args)
        masks = prediction["masks"]
        union = (
            np.any(masks, axis=0)
            if masks.shape[0]
            else np.zeros(depth.shape, dtype=bool)
        )
        valid = (
            union
            & np.isfinite(depth)
            & (depth > 0.0)
            & np.all(np.isfinite(points_camera), axis=2)
        )
        camera_points = points_camera[valid].astype(np.float32)
        world_points = points_world[valid].astype(np.float32)

        np.save(output / "union_mask.npy", union)
        np.save(output / "valid_3d_mask.npy", valid)
        np.save(output / "points_camera.npy", camera_points)
        np.save(output / "points_world.npy", world_points)
        save_overlay(output / "overlay.png", rgb, union)
        write_colored_cloud(output / "points_world.ply", world_points, (40, 200, 90))

        truth = gt.points_in_tool_mask(tool_pose, scene, points_world, margin_m=0.0)
        agreement = agreement_with_ground_truth(union, truth)

        checks = {
            "sam3_found_something": bool(masks.shape[0] > 0),
            "mask_has_valid_depth": bool(camera_points.shape[0] > 0),
            "enough_points_for_a_grasp_proposal": bool(camera_points.shape[0] >= 100),
        }
        report = {
            "status": "success" if all(checks.values()) else "failed_checks",
            "reference": "Hugging Face Transformers SAM3, text-prompted",
            "capture": str(args.capture),
            "prompt": prompt,
            "model_id": args.model_id,
            "thresholds": {
                "score": args.score_threshold,
                "mask": args.mask_threshold,
            },
            "instances": [
                {
                    "index": index,
                    "score": round(float(prediction["scores"][index]), 4),
                    "box_xyxy_px": [
                        round(float(v), 2) for v in prediction["boxes"][index]
                    ],
                    "mask_pixels": int(np.count_nonzero(masks[index])),
                }
                for index in range(masks.shape[0])
            ],
            "points": {
                "valid_3d": int(camera_points.shape[0]),
                "frame_camera": "OpenCV optical, as the capture saved it",
            },
            "agreement_with_ground_truth": agreement,
            "inspect": {
                "overlay": str(output / "overlay.png"),
                "point_cloud": str(output / "points_world.ply"),
            },
            "automatic_checks": checks,
            "next_step": (
                f"python scripts/propose_grasps.py --capture {args.capture} "
                f"--segmentation {output}"
            ),
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(checks, indent=2), flush=True)
        print(
            f"points: {camera_points.shape[0]}  "
            f"IoU vs truth: {agreement['intersection_over_union']}  "
            f"(spilled {agreement['spilled_pixels']}, missed {agreement['missed_pixels']})",
            flush=True,
        )
        print(f"SEGMENT {report['status'].upper()} -> {report_path}", flush=True)
        return 0 if report["status"] == "success" else 2
    except Exception as error:
        import traceback

        report_path.write_text(
            json.dumps(
                {
                    "status": "failure",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        print(f"failure report: {report_path}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
