from __future__ import annotations

try:
    import bpy as _bpy
except ModuleNotFoundError:
    _bpy = None

if _bpy is not None:
    from .operators import register, unregister
else:
    def register() -> None:
        raise RuntimeError("The Chromehounds extension can only be registered in Blender")


    def unregister() -> None:
        pass
