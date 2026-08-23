"""Frame handling around NVIDIA's GraspGenX server.

Grasp generation itself stays in the official server; this module only shapes
its input, converts its output into the frames the rest of the project uses,
and checks that the conversion is right.

A wrong frame here does not look broken.  It produces plausible poses in
plausible places that are all offset the same way, and the first sign of it is
a robot closing its fingers on air.  So the conversion is verified against the
points the grasps were proposed from, not merely asserted.

Ported from jikkenn1, where this ran against the same server.
"""

from __future__ import annotations

import numpy as np

#: From GraspGenX's own ``end2end/robots/franka_panda.yaml``, copied as data.
#: GraspGenX closes along +X; the Panda's ``panda_hand`` closes along +Y, so the
#: two frames differ by a quarter turn about the shared approach axis.
T_GRASP_PANDA_HAND = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def prepare_scene_point_cloud(
    points_camera: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Shape one organized point map for ``infer_scene_pc``.

    Pixels without a finite point stay non-finite -- the server ignores them --
    but their instance label is cleared, so the count reported here is exactly
    what the server sees as instance 1.
    """
    points = np.asarray(points_camera)
    selected = np.asarray(mask, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(f"points_camera must be HxWx3, got {points.shape}")
    if selected.shape != points.shape[:2]:
        raise ValueError(
            f"mask shape {selected.shape} does not match the point map {points.shape[:2]}"
        )
    valid = np.all(np.isfinite(points), axis=2)
    instance = (selected & valid).astype(np.int32)
    return points.astype(np.float32, copy=False), instance, int(instance.sum())


def transform_grasp_poses(
    grasps_camera: np.ndarray, T_world_camera: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert GraspGenX poses into world poses and ``panda_hand`` poses."""
    grasps = np.asarray(grasps_camera, dtype=np.float64)
    transform = np.asarray(T_world_camera, dtype=np.float64)
    if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
        raise ValueError(f"grasps_camera must be (N, 4, 4), got {grasps.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"T_world_camera must be 4x4, got {transform.shape}")
    if not np.all(np.isfinite(grasps)) or not np.all(np.isfinite(transform)):
        raise ValueError("grasp poses and T_world_camera must be finite")
    grasps_world = np.einsum("ij,njk->nik", transform, grasps)
    panda_hand_world = np.einsum("nij,jk->nik", grasps_world, T_GRASP_PANDA_HAND)
    return grasps_world, panda_hand_world


def pose_quality(poses: np.ndarray) -> dict:
    """Residuals of the returned transforms, reported rather than repaired."""
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (N, 4, 4), got {values.shape}")
    if values.shape[0] == 0:
        return {
            "finite": True,
            "max_rotation_orthogonality_error": 0.0,
            "max_rotation_determinant_error": 0.0,
            "max_homogeneous_row_error": 0.0,
        }
    rotations = values[:, :3, :3]
    orthogonality = np.matmul(np.transpose(rotations, (0, 2, 1)), rotations)
    return {
        "finite": bool(np.all(np.isfinite(values))),
        "max_rotation_orthogonality_error": float(
            np.max(np.abs(orthogonality - np.eye(3)))
        ),
        "max_rotation_determinant_error": float(
            np.max(np.abs(np.linalg.det(rotations) - 1.0))
        ),
        "max_homogeneous_row_error": float(
            np.max(np.abs(values[:, 3, :] - np.array([0.0, 0.0, 0.0, 1.0])))
        ),
    }


def fingertip_agreement(
    hand_poses: np.ndarray,
    object_points_world: np.ndarray,
    *,
    fingertip_depth_m: float,
    tolerance_m: float = 0.03,
) -> dict:
    """Do the converted fingertips land on the points the grasps came from?

    This is the check that catches a wrong frame conversion.  Every proposal was
    made *for* this point cloud, so after conversion the fingers must close near
    it.  A systematic offset shows up here as a large median distance, long
    before it shows up as a robot gripping thin air.
    """
    poses = np.asarray(hand_poses, dtype=np.float64)
    points = np.asarray(object_points_world, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"hand_poses must be (N, 4, 4), got {poses.shape}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"object_points_world must be (M, 3), got {points.shape}")
    if poses.shape[0] == 0 or points.shape[0] == 0:
        return {"checked": 0, "passed": False, "reason": "nothing to compare"}

    fingertips = poses[:, :3, 3] + poses[:, :3, 2] * float(fingertip_depth_m)
    distances = np.empty(len(fingertips))
    for index, fingertip in enumerate(fingertips):
        distances[index] = np.min(np.linalg.norm(points - fingertip, axis=1))
    median = float(np.median(distances))
    return {
        "checked": int(len(fingertips)),
        "median_distance_m": round(median, 5),
        "maximum_distance_m": round(float(np.max(distances)), 5),
        "tolerance_m": tolerance_m,
        "passed": median <= tolerance_m,
        "means": (
            "the converted fingertips sit on the segmented object"
            if median <= tolerance_m
            else "the fingertips are systematically off the object; suspect the "
            "grasp-to-hand frame conversion before blaming the proposer"
        ),
    }
