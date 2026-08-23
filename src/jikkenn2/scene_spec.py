"""Single source of truth for the Phase 0 scene, with machine-checkable geometry.

Every metric value used by ``build_scene_usd.py`` lives here, and every claim
made about that layout ("the camera sees the whole table", "the handover pose is
reachable", "the robot does not shadow the workspace") is expressed as a check
that runs without Isaac Sim, CUDA or a GPU.

Conventions
-----------
* World frame: the Panda mounting plane and the tabletop are both ``z = 0``.
* The human stands on the ``-y`` side; the camera is on the opposite ``+y`` side
  so a reaching arm never occludes it.
* Camera axes follow the OpenCV optical convention: ``+x`` image right,
  ``+y`` image down, ``+z`` forward.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


WORLD_UP = np.array([0.0, 0.0, 1.0])

# Franka Emika Panda, from the official datasheet.  Used only as an outer bound;
# the authoritative reachability answer comes from cuRobo IK in
# ``reachability_map.py``.
PANDA_MAX_REACH_M = 0.855


@dataclass(frozen=True)
class ToolPart:
    """One labelled part of the proxy tool, in the tool's local frame."""

    name: str
    size_m: tuple[float, float, float]
    center_m: tuple[float, float, float]
    axis_local: tuple[float, float, float]

    def aabb_local_m(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        center = np.asarray(self.center_m, dtype=np.float64)
        half = 0.5 * np.asarray(self.size_m, dtype=np.float64)
        return tuple(center - half), tuple(center + half)


@dataclass(frozen=True)
class SceneSpec:
    """Metric layout of the Phase 0 tabletop scene."""

    # --- robot -----------------------------------------------------------
    robot_base_position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_body_radius_m: float = 0.09
    # Bounding cylinder of the *home* configuration, which is the pose the
    # camera captures.  A fully extended Panda reaches roughly 1.19 m and does
    # leave the top of the frame; that is acceptable, because an arm outside the
    # image contributes nothing to the depth and therefore needs no masking.
    robot_body_height_m: float = 1.00

    # --- room and table --------------------------------------------------
    ground_z_m: float = -1.05
    table_center_m: tuple[float, float, float] = (0.55, 0.0, -0.025)
    table_size_m: tuple[float, float, float] = (0.70, 1.20, 0.05)

    # --- camera (opposite side from the human) ---------------------------
    camera_position_m: tuple[float, float, float] = (0.50, 1.30, 1.60)
    camera_target_m: tuple[float, float, float] = (0.50, 0.00, 0.05)
    camera_resolution_px: tuple[int, int] = (640, 480)
    camera_horizontal_fov_deg: float = 69.0
    camera_clip_m: tuple[float, float] = (0.05, 5.0)

    # --- task points -----------------------------------------------------
    handover_position_m: tuple[float, float, float] = (0.40, -0.45, 0.35)
    human_point_m: tuple[float, float, float] = (0.45, -1.10, 0.35)
    tool_initial_position_m: tuple[float, float, float] = (0.45, 0.15, 0.0)

    # --- working band (provisional; reachability_map.py refines it) ------
    reach_min_m: float = 0.30
    reach_max_m: float = 0.65
    handover_reach_max_m: float = 0.75

    # --- proxy tool ------------------------------------------------------
    tool_parts: tuple[ToolPart, ...] = field(
        default_factory=lambda: (
            ToolPart("head", (0.060, 0.045, 0.045), (0.030, 0.0, 0.0), (1.0, 0.0, 0.0)),
            ToolPart("handle", (0.140, 0.028, 0.028), (-0.070, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        )
    )
    danger_part_name: str = "head"
    safe_part_name: str = "handle"
    grasp_part_name: str = "head"

    # ------------------------------------------------------------------
    # derived table geometry
    # ------------------------------------------------------------------
    @property
    def table_top_z_m(self) -> float:
        return self.table_center_m[2] + 0.5 * self.table_size_m[2]

    @property
    def table_bottom_z_m(self) -> float:
        return self.table_center_m[2] - 0.5 * self.table_size_m[2]

    @property
    def table_min_xy_m(self) -> tuple[float, float]:
        return (
            self.table_center_m[0] - 0.5 * self.table_size_m[0],
            self.table_center_m[1] - 0.5 * self.table_size_m[1],
        )

    @property
    def table_max_xy_m(self) -> tuple[float, float]:
        return (
            self.table_center_m[0] + 0.5 * self.table_size_m[0],
            self.table_center_m[1] + 0.5 * self.table_size_m[1],
        )

    def table_top_corners_m(self) -> np.ndarray:
        (min_x, min_y), (max_x, max_y) = self.table_min_xy_m, self.table_max_xy_m
        z = self.table_top_z_m
        return np.array(
            [[min_x, min_y, z], [max_x, min_y, z], [min_x, max_y, z], [max_x, max_y, z]]
        )

    def point_is_over_table(self, point_m) -> bool:
        point = np.asarray(point_m, dtype=np.float64)
        (min_x, min_y), (max_x, max_y) = self.table_min_xy_m, self.table_max_xy_m
        return bool(min_x <= point[0] <= max_x and min_y <= point[1] <= max_y)

    def part(self, name: str) -> ToolPart:
        for candidate in self.tool_parts:
            if candidate.name == name:
                return candidate
        raise KeyError(f"unknown tool part: {name}")

    # ------------------------------------------------------------------
    # camera model
    # ------------------------------------------------------------------
    @property
    def camera_vertical_fov_deg(self) -> float:
        width, height = self.camera_resolution_px
        half_h = np.deg2rad(self.camera_horizontal_fov_deg) / 2.0
        return float(np.rad2deg(2.0 * np.arctan(np.tan(half_h) * height / width)))

    def camera_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(forward, right, down)`` unit vectors in OpenCV optical order."""
        position = np.asarray(self.camera_position_m, dtype=np.float64)
        target = np.asarray(self.camera_target_m, dtype=np.float64)
        forward = target - position
        norm = float(np.linalg.norm(forward))
        if norm <= 1e-9:
            raise ValueError("camera position and target must differ")
        forward = forward / norm
        if abs(float(np.dot(forward, WORLD_UP))) > 0.999:
            raise ValueError("camera looks straight along world up; pick another target")
        right = np.cross(forward, WORLD_UP)
        right = right / float(np.linalg.norm(right))
        down = np.cross(forward, right)
        return forward, right, down

    def camera_angles_deg(self, point_m) -> tuple[float, float, float]:
        """Return ``(horizontal_deg, vertical_deg, depth_m)`` for a world point.

        Angles are signed offsets from the optical axis; ``depth_m`` is the
        planar depth used by pinhole back-projection.
        """
        forward, right, down = self.camera_axes()
        offset = np.asarray(point_m, dtype=np.float64) - np.asarray(
            self.camera_position_m, dtype=np.float64
        )
        depth = float(np.dot(offset, forward))
        if depth <= 0.0:
            return (180.0, 180.0, depth)
        horizontal = float(np.rad2deg(np.arctan2(float(np.dot(offset, right)), depth)))
        vertical = float(np.rad2deg(np.arctan2(float(np.dot(offset, down)), depth)))
        return horizontal, vertical, depth

    def point_is_in_view(self, point_m, *, margin_deg: float = 0.0) -> bool:
        horizontal, vertical, depth = self.camera_angles_deg(point_m)
        near, far = self.camera_clip_m
        return bool(
            abs(horizontal) <= 0.5 * self.camera_horizontal_fov_deg - margin_deg
            and abs(vertical) <= 0.5 * self.camera_vertical_fov_deg - margin_deg
            and near <= depth <= far
        )

    # ------------------------------------------------------------------
    # robot geometry
    # ------------------------------------------------------------------
    def robot_height_fully_in_view_m(self, *, resolution_m: float = 0.005) -> float:
        """Highest point straight above the base that still fits in the frame.

        Reported so the margin between the robot envelope and the top edge of
        the image is a number, not an assumption.
        """
        base = np.asarray(self.robot_base_position_m, dtype=np.float64)
        low, high = 0.0, 3.0
        if not self.point_is_in_view(base):
            return 0.0
        while high - low > resolution_m:
            middle = 0.5 * (low + high)
            probe = base + np.array([0.0, 0.0, middle])
            if self.point_is_in_view(probe):
                low = middle
            else:
                high = middle
        return float(low)

    def distance_from_robot_base_m(self, point_m) -> float:
        offset = np.asarray(point_m, dtype=np.float64) - np.asarray(
            self.robot_base_position_m, dtype=np.float64
        )
        return float(np.linalg.norm(offset))

    def robot_shadow_on_table_top(self, *, samples: int = 64) -> np.ndarray:
        """Where the camera's view of the tabletop is blocked by the robot body.

        The robot is bounded by an upright cylinder.  Each sampled point on that
        cylinder is projected along its camera ray down to the tabletop plane;
        the returned points are where the tabletop is hidden behind the robot.
        Points that never reach the plane in front of the camera are dropped.
        """
        camera = np.asarray(self.camera_position_m, dtype=np.float64)
        base = np.asarray(self.robot_base_position_m, dtype=np.float64)
        plane_z = self.table_top_z_m

        angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
        heights = np.linspace(0.0, self.robot_body_height_m, 9)
        ring = np.stack(
            [
                self.robot_body_radius_m * np.cos(angles),
                self.robot_body_radius_m * np.sin(angles),
                np.zeros_like(angles),
            ],
            axis=1,
        )
        surface = np.concatenate(
            [ring + base + np.array([0.0, 0.0, height]) for height in heights], axis=0
        )

        direction = surface - camera
        # Only rays that descend can meet the tabletop plane below the camera.
        descending = direction[:, 2] < -1e-9
        direction = direction[descending]
        if direction.shape[0] == 0:
            return np.empty((0, 3))
        travel = (plane_z - camera[2]) / direction[:, 2]
        forward_only = travel > 1.0  # beyond the robot itself
        return camera + direction[forward_only] * travel[forward_only, None]

    # ------------------------------------------------------------------
    # report
    # ------------------------------------------------------------------
    def validation_report(self) -> dict:
        corners = self.table_top_corners_m()
        corner_angles = [self.camera_angles_deg(corner) for corner in corners]
        robot_probes = np.array(
            [
                self.robot_base_position_m,
                (
                    self.robot_base_position_m[0],
                    self.robot_base_position_m[1],
                    self.robot_base_position_m[2] + self.robot_body_height_m,
                ),
            ]
        )
        shadow = self.robot_shadow_on_table_top()
        shadow_on_table = [point for point in shadow if self.point_is_over_table(point)]

        tool_center = np.asarray(self.tool_initial_position_m, dtype=np.float64).copy()
        tool_center[2] = self.table_top_z_m + 0.5 * max(
            part.size_m[2] for part in self.tool_parts
        )
        handover_distance = self.distance_from_robot_base_m(self.handover_position_m)
        tool_distance = self.distance_from_robot_base_m(tool_center)

        handover_to_human = np.asarray(self.human_point_m, dtype=np.float64) - np.asarray(
            self.handover_position_m, dtype=np.float64
        )
        handover_to_human /= float(np.linalg.norm(handover_to_human))

        checks = {
            "robot_mount_matches_tabletop": abs(
                self.robot_base_position_m[2] - self.table_top_z_m
            )
            <= 1e-9,
            "floor_is_below_table": self.ground_z_m < self.table_bottom_z_m,
            "table_clears_robot_body": (
                self.table_min_xy_m[0] - self.robot_body_radius_m >= 0.05
            ),
            "camera_sees_every_table_corner": all(
                self.point_is_in_view(corner) for corner in corners
            ),
            "camera_frames_robot_home_envelope": all(
                self.point_is_in_view(probe) for probe in robot_probes
            ),
            "camera_sees_handover_pose": self.point_is_in_view(self.handover_position_m),
            "camera_sees_human_point": self.point_is_in_view(self.human_point_m),
            "robot_does_not_shadow_the_table": len(shadow_on_table) == 0,
            "tool_start_is_over_the_table": self.point_is_over_table(tool_center),
            "tool_start_is_in_working_band": (
                self.reach_min_m <= tool_distance <= self.reach_max_m
            ),
            "handover_is_within_panda_reach": handover_distance <= self.handover_reach_max_m,
            "handover_is_below_datasheet_reach": handover_distance < PANDA_MAX_REACH_M,
            "human_is_on_the_negative_y_side": self.human_point_m[1] < self.table_min_xy_m[1],
            "camera_is_on_the_opposite_side_from_the_human": (
                self.camera_position_m[1] > self.table_max_xy_m[1]
            ),
            "handover_faces_the_human_along_minus_y": handover_to_human[1] < -0.8,
        }
        # NumPy comparisons return np.bool_, which json.dumps refuses.
        checks = {name: bool(value) for name, value in checks.items()}

        return {
            "status": "success" if all(checks.values()) else "failure",
            "convention": {
                "world": "Panda mounting plane and tabletop are both z=0",
                "camera": "OpenCV optical, +x right, +y down, +z forward",
                "human_side": "-y",
                "camera_side": "+y",
            },
            "layout": {
                key: value
                for key, value in asdict(self).items()
                if key != "tool_parts"
            },
            "tool_parts": [asdict(part) for part in self.tool_parts],
            "derived": {
                "table_top_z_m": self.table_top_z_m,
                "table_min_xy_m": list(self.table_min_xy_m),
                "table_max_xy_m": list(self.table_max_xy_m),
                "camera_vertical_fov_deg": self.camera_vertical_fov_deg,
                "camera_half_fov_deg": [
                    0.5 * self.camera_horizontal_fov_deg,
                    0.5 * self.camera_vertical_fov_deg,
                ],
                "table_corner_angles_deg_and_depth_m": [
                    [round(h, 2), round(v, 2), round(d, 3)] for h, v, d in corner_angles
                ],
                "robot_probe_angles_deg_and_depth_m": [
                    [round(h, 2), round(v, 2), round(d, 3)]
                    for h, v, d in (self.camera_angles_deg(p) for p in robot_probes)
                ],
                "handover_distance_from_base_m": round(handover_distance, 4),
                "tool_start_distance_from_base_m": round(tool_distance, 4),
                "tool_start_center_m": [round(float(v), 4) for v in tool_center],
                "robot_shadow_samples_on_table": len(shadow_on_table),
                "robot_height_fully_in_view_m": round(self.robot_height_fully_in_view_m(), 3),
                "robot_home_envelope_height_m": self.robot_body_height_m,
            },
            "automatic_checks": checks,
        }


DEFAULT_SCENE = SceneSpec()
