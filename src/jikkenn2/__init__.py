"""jikkenn2: danger-part-aware tool handover in Isaac Sim.

Phase 0 uses no perception at all.  Object poses come from the simulator's
ground truth, and the map comes from cuRobo's official ESDF.  Perception is
substituted in later, one layer per phase, always against the previous phase as
a baseline.
"""

from jikkenn2.scene_spec import DEFAULT_SCENE, SceneSpec, ToolPart

__all__ = ["DEFAULT_SCENE", "SceneSpec", "ToolPart"]
