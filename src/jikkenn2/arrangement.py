"""Reading, checking and restoring hand-made object placements.

An *arrangement* is where the human put the movable objects on the table.  It is
the reproducible half of the experiment: a person drags things in the Isaac Sim
GUI once, the poses are saved here, and every later run replays them without a
GUI.

Only USD is needed, so all of this is testable without Isaac Sim or a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np

from jikkenn2.scene_spec import SceneSpec


SCHEMA_VERSION = 1

# Prims under these roots are the ones a person is expected to move.  Everything
# else (table, robot, camera, markers) is structure and comes from scene_spec.
MOVABLE_ROOTS = ("/World/Tools", "/World/Obstacles")

ARRANGEMENT_PATTERN = re.compile(r"^arr_(\d+)\.json$")


@dataclass(frozen=True)
class PlacedObject:
    """One movable prim and where the human left it."""

    prim_path: str
    position_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]

    def as_dict(self) -> dict:
        return {
            "prim_path": self.prim_path,
            "position_m": [float(v) for v in self.position_m],
            "orientation_wxyz": [float(v) for v in self.orientation_wxyz],
        }


def iter_movable_prim_paths(stage) -> list[str]:
    """Return the direct children of every movable root, in stage order."""
    paths: list[str] = []
    for root in MOVABLE_ROOTS:
        prim = stage.GetPrimAtPath(root)
        if not prim.IsValid():
            continue
        paths.extend(child.GetPath().pathString for child in prim.GetChildren())
    return paths


def read_pose(stage, prim_path: str) -> PlacedObject:
    """Read a prim's world pose, whatever xform ops it happens to use."""
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise KeyError(f"prim not found: {prim_path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    transform = Gf.Transform(matrix)
    translation = transform.GetTranslation()
    quaternion = transform.GetRotation().GetQuat()
    imaginary = quaternion.GetImaginary()
    return PlacedObject(
        prim_path=prim_path,
        position_m=(
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        ),
        orientation_wxyz=(
            float(quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ),
    )


def check_placement(scene: SceneSpec, position_m) -> dict:
    """Report whether one object is somewhere the robot can actually work.

    This is a cheap geometric screen, not a reachability answer.  The
    authoritative answer comes from cuRobo IK in ``reachability_map.py``; this
    exists so a placement mistake is caught the moment it is saved.
    """
    position = np.asarray(position_m, dtype=np.float64)
    distance = scene.distance_from_robot_base_m(position)
    return {
        "distance_from_base_m": round(float(distance), 4),
        "over_the_table": scene.point_is_over_table(position),
        "in_working_band": bool(scene.reach_min_m <= distance <= scene.reach_max_m),
        "above_the_tabletop": bool(position[2] >= scene.table_top_z_m - 1e-6),
        "visible_to_the_camera": scene.point_is_in_view(position),
    }


def collect_arrangement(stage, scene: SceneSpec, *, source_stage: str) -> dict:
    """Snapshot every movable prim, with a placement screen for each."""
    objects = [read_pose(stage, path) for path in iter_movable_prim_paths(stage)]
    if not objects:
        raise ValueError(
            "no movable prims found; expected children under " + ", ".join(MOVABLE_ROOTS)
        )
    entries = []
    for placed in objects:
        entry = placed.as_dict()
        entry["placement"] = check_placement(scene, placed.position_m)
        entries.append(entry)

    problems = [
        entry["prim_path"]
        for entry in entries
        if not (
            entry["placement"]["over_the_table"]
            and entry["placement"]["in_working_band"]
            and entry["placement"]["above_the_tabletop"]
        )
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "source_stage": source_stage,
        "frame": "Isaac world; the Panda mounting plane and tabletop are z=0",
        "settled": False,
        "note": (
            "Poses are as authored in the GUI. capture_scene.py settles physics "
            "before capturing and records the settled poses with the trial."
        ),
        "objects": entries,
        "objects_outside_the_working_area": problems,
        "status": "success" if not problems else "placement_warning",
    }


def next_arrangement_path(directory: str | Path) -> Path:
    """Return the next free ``arr_NNN.json`` in a directory."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    used = []
    for path in folder.glob("arr_*.json"):
        match = ARRANGEMENT_PATTERN.match(path.name)
        if match:
            used.append(int(match.group(1)))
    return folder / f"arr_{(max(used) + 1) if used else 1:03d}.json"


def save_arrangement(arrangement: dict, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(arrangement, indent=2) + "\n", encoding="utf-8")
    return destination


def load_arrangement(path: str | Path) -> dict:
    """Load an arrangement, refusing anything this code cannot interpret."""
    arrangement = json.loads(Path(path).read_text(encoding="utf-8"))
    version = arrangement.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"arrangement schema {version!r} is not supported (expected {SCHEMA_VERSION})"
        )
    objects = arrangement.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("arrangement contains no objects")
    for entry in objects:
        if not isinstance(entry, dict) or "prim_path" not in entry:
            raise ValueError("arrangement object entry is malformed")
        if len(entry.get("position_m", ())) != 3:
            raise ValueError(f"{entry.get('prim_path')}: position_m must have 3 values")
        if len(entry.get("orientation_wxyz", ())) != 4:
            raise ValueError(
                f"{entry.get('prim_path')}: orientation_wxyz must have 4 values"
            )
    return arrangement


def apply_arrangement(stage, arrangement: dict) -> list[str]:
    """Put every recorded object back where it was; returns the prim paths set.

    Poses are authored as a translate plus an orient op at the precision each
    prim already uses, so a referenced asset's own ops are never fought.
    """
    from pxr import Gf, UsdGeom

    applied = []
    for entry in arrangement["objects"]:
        prim = stage.GetPrimAtPath(entry["prim_path"])
        if not prim.IsValid():
            raise KeyError(f"arrangement refers to a missing prim: {entry['prim_path']}")
        xform = UsdGeom.Xformable(prim)
        existing = {op.GetOpType(): op for op in xform.GetOrderedXformOps()}

        translate = existing.get(UsdGeom.XformOp.TypeTranslate)
        if translate is None:
            translate = xform.AddTranslateOp()
        position = entry["position_m"]
        if translate.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            translate.Set(Gf.Vec3d(*(float(v) for v in position)))
        else:
            translate.Set(Gf.Vec3f(*(float(v) for v in position)))

        orient = existing.get(UsdGeom.XformOp.TypeOrient)
        if orient is None:
            orient = xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        w, x, y, z = (float(v) for v in entry["orientation_wxyz"])
        if orient.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            orient.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        else:
            orient.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        applied.append(entry["prim_path"])
    return applied


def describe_placement(entry: dict) -> str:
    """One terminal line per object, for the interactive save confirmation."""
    placement = entry["placement"]
    flags = []
    if not placement["over_the_table"]:
        flags.append("off table")
    if not placement["in_working_band"]:
        flags.append(f"outside reach band ({placement['distance_from_base_m']:.2f} m)")
    if not placement["above_the_tabletop"]:
        flags.append("below the tabletop")
    if not placement["visible_to_the_camera"]:
        flags.append("not in camera view")
    verdict = "OK" if not flags else "WARN: " + ", ".join(flags)
    name = entry["prim_path"].rsplit("/", 1)[-1]
    position = entry["position_m"]
    return (
        f"  {name:<14} "
        f"({position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f})  {verdict}"
    )
