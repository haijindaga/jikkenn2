"""Choosing which proposed grasp to use.

GraspGenX proposes where a gripper could close on a point cloud; it has no idea
which end of a knife is the blade.  Deciding that is the whole claim, so it
happens here, explicitly, and on candidates that came from perception rather
than from the tool's true pose.

Phase 2 still asks the ground truth *which part is dangerous* -- that is what
Phase 3 hands to SAM3.  What changes here is where the grasp poses come from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jikkenn2 import ground_truth as gt
from jikkenn2.reachability import PANDA_FINGERTIP_DEPTH_M
from jikkenn2.scene_spec import SceneSpec

#: A grasp is "from above" when the hand's approach axis points down.  Used only
#: to break ties between candidates the proposer scored equally.
STRAIGHT_DOWN = np.array([0.0, 0.0, -1.0])


@dataclass(frozen=True)
class GraspCandidate:
    """One proposed ``panda_hand`` pose, with whatever score came with it."""

    index: int
    hand_pose: np.ndarray
    score: float

    def fingertip_m(self, depth_m: float = PANDA_FINGERTIP_DEPTH_M) -> np.ndarray:
        pose = np.asarray(self.hand_pose, dtype=np.float64)
        return pose[:3, 3] + pose[:3, 2] * float(depth_m)

    def downwardness(self) -> float:
        """1.0 straight down, 0.0 horizontal, negative pointing up."""
        pose = np.asarray(self.hand_pose, dtype=np.float64)
        return float(np.dot(pose[:3, 2], STRAIGHT_DOWN))


def candidates_from_arrays(hand_poses, scores) -> list[GraspCandidate]:
    poses = np.asarray(hand_poses, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"hand_poses must be (N, 4, 4), got {poses.shape}")
    if values.shape[0] != poses.shape[0]:
        raise ValueError(
            f"{poses.shape[0]} poses but {values.shape[0]} scores"
        )
    return [
        GraspCandidate(index=index, hand_pose=poses[index], score=float(values[index]))
        for index in range(poses.shape[0])
    ]


def rank_candidates(
    candidates: list[GraspCandidate],
    tool_pose: np.ndarray,
    scene: SceneSpec,
    *,
    minimum_downwardness: float = 0.0,
    fingertip_depth_m: float = PANDA_FINGERTIP_DEPTH_M,
) -> dict:
    """Keep the grasps that land on the part to be grasped, best first.

    Ranking is by the proposer's own score, with how far the hand points
    downward breaking ties: two grasps the proposer likes equally are not
    equally easy for an arm mounted on a table.
    """
    kept = []
    rejected = []
    for candidate in candidates:
        fingertip = candidate.fingertip_m(fingertip_depth_m)
        part = gt.part_containing(tool_pose, scene, fingertip)
        downward = candidate.downwardness()
        entry = {
            "index": candidate.index,
            "score": round(candidate.score, 5),
            "part": part,
            "downwardness": round(downward, 4),
            "fingertip_m": [round(float(v), 5) for v in fingertip],
        }
        if part != scene.grasp_part_name:
            entry["rejected_because"] = (
                f"the fingers close on {part!r}, not {scene.grasp_part_name!r}"
            )
            rejected.append(entry)
        elif downward < minimum_downwardness:
            entry["rejected_because"] = (
                f"approach is not downward enough ({downward:.2f} < "
                f"{minimum_downwardness})"
            )
            rejected.append(entry)
        else:
            kept.append((candidate, entry))

    kept.sort(key=lambda pair: (-pair[1]["score"], -pair[1]["downwardness"]))
    for rank, (_, entry) in enumerate(kept):
        entry["rank"] = rank
    return {
        "ordered": [candidate for candidate, _ in kept],
        "kept": [entry for _, entry in kept],
        "rejected": rejected,
        "counts": {
            "proposed": len(candidates),
            "on_the_intended_part": len(kept),
            "rejected": len(rejected),
        },
        "intended_part": scene.grasp_part_name,
    }


def rejection_summary(ranking: dict) -> dict:
    """Why the proposals were thrown out, grouped, for the report."""
    by_part: dict[str, int] = {}
    for entry in ranking["rejected"]:
        key = str(entry["part"])
        by_part[key] = by_part.get(key, 0) + 1
    return {
        "rejected_by_part": by_part,
        "note": (
            "a proposal landing on the safe part is not a proposer failure; it "
            "is exactly the choice this project exists to make"
        ),
    }
