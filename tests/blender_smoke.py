from __future__ import annotations

import os
import json
import sys
import tempfile
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chromehounds_importer.importer import ImportSettings, import_asset_data
from chromehounds_importer.blender_data import TextureLibrary, build_material
from chromehounds_importer.model import MaterialParam, ModelMaterial
from chromehounds_importer.source import DirectorySource


def main() -> None:
    game_root_value = os.environ.get("CHROMEHOUNDS_GAME_ROOT")
    if not game_root_value:
        raise RuntimeError("CHROMEHOUNDS_GAME_ROOT is required")
    game_root = Path(game_root_value)

    part = game_root / "parts" / "AX_AM001.bnd"
    part_result = import_asset_data(
        part.read_bytes(),
        str(part),
        bpy.context,
        ImportSettings(),
    )
    assert part_result.models == 1
    assert part_result.meshes == 1
    assert part_result.images == 3
    armature = bpy.data.objects["AX_AM001"]
    assert armature.type == "ARMATURE"
    assert len(armature.data.bones) == 2
    mesh = bpy.data.objects["AX_AM001_Mesh_000"]
    assert len(mesh.data.vertices) == 748
    assert len(mesh.data.polygons) == 424
    assert len(mesh.data.uv_layers) == 4
    material = mesh.data.materials[0]
    assert material.get("chromehounds_alpha_mapping") == "OPAQUE"
    assert not material.use_transparency_overlap
    assert not material.use_transparent_shadow
    assert not any(
        link.to_node.bl_idname == "ShaderNodeBsdfPrincipled"
        and link.to_socket.name == "Alpha"
        for link in material.node_tree.links
    )
    for image_name in ("AX_AM001", "AX_AM001_b", "AX_AM001_c"):
        image = bpy.data.images[image_name]
        assert image.use_fake_user
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        rgb = pixels.reshape(-1, image.channels)[:, :3]
        assert float(np.ptp(rgb)) > 0.01, image_name
    additive_material = build_material(
        ModelMaterial(
            name="Additive Smoke",
            shader="MDL_NoLight",
            flags=(0, 0, 0),
            params=(
                MaterialParam(kind=0, name="g_MatAlphaBlend", value=3),
                MaterialParam(
                    kind=4,
                    name="g_MatDiffCol",
                    value=(1.0, 1.0, 1.0, 1.0),
                ),
            ),
        ),
        TextureLibrary(),
    )
    assert additive_material.get("chromehounds_alpha_mapping") == "ADDITIVE"
    assert additive_material.surface_render_method == "BLENDED"
    additive_surface_link = next(
        link
        for link in additive_material.node_tree.links
        if link.to_node.bl_idname == "ShaderNodeOutputMaterial"
        and link.to_socket.name == "Surface"
    )
    assert additive_surface_link.from_node.bl_idname == "ShaderNodeAddShader"

    character = game_root / "chr" / "c0101.bnd"
    character_result = import_asset_data(
        character.read_bytes(),
        str(character),
        bpy.context,
        ImportSettings(import_textures=False),
    )
    assert character_result.actions == 4
    assert bpy.context.scene.render.fps == 30
    character_armature = bpy.data.objects["c0101"]
    assert len(character_armature.data.bones) == 23
    assert all(len(action.layers) == 1 for action in bpy.data.actions)
    assert all(len(action.slots) == 1 for action in bpy.data.actions)
    assert all(action.use_fake_user for action in bpy.data.actions)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for original in (
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and obj.name.startswith("c0101")
    ):
        evaluated = original.evaluated_get(depsgraph)
        max_delta = max(
            (
                evaluated.data.vertices[index].co - vertex.co
            ).length
            for index, vertex in enumerate(original.data.vertices)
        )
        assert max_delta < 1e-4
    character_armature.animation_data.action = bpy.data.actions["c0101_Clip_000"]
    bpy.context.scene.frame_set(60)
    root_translation = character_armature.pose.bones["master"].matrix.translation
    assert root_translation.y > 5.0
    assert abs(root_translation.z) < 1e-4
    character_armature.animation_data.action = None
    bpy.context.scene.frame_set(0)

    relative_map = "mapdata/map/m01_001/m01_001.msb"
    map_path = game_root.joinpath(*relative_map.split("/"))
    map_data = map_path.read_bytes()
    map_result = import_asset_data(
        map_data,
        relative_map,
        bpy.context,
        ImportSettings(
            import_textures=False,
            create_armatures=False,
            max_map_instances=100,
            map_placement_mode="LINKED_OBJECTS",
        ),
        source=DirectorySource(game_root),
        virtual_path=relative_map,
    )
    assert map_result.instances == 100
    assert map_result.models >= 54
    placement_roots = [
        obj
        for obj in bpy.data.objects
        if obj.get("chromehounds_map_placement", False)
        and obj.get("chromehounds_placement_mode") == "LINKED_OBJECTS"
    ]
    assert len(placement_roots) == 100
    assert not any(
        obj.instance_type == "COLLECTION" for obj in placement_roots
    )
    assert max(abs(obj.location.x) for obj in placement_roots) > 10.0
    placement_meshes = [
        child
        for root in placement_roots
        for child in root.children
        if child.type == "MESH"
    ]
    assert placement_meshes
    assert all(mesh.data.materials for mesh in placement_meshes)
    assert all(mesh.data.users > 1 for mesh in placement_meshes)
    assert all(
        not material.use_backface_culling
        for mesh in placement_meshes
        for material in mesh.data.materials
    )
    assert all(
        not mesh.show_transparent
        for mesh in placement_meshes
        if all(
            material.get("chromehounds_alpha_mapping") == "OPAQUE"
            for material in mesh.data.materials
        )
    )
    legacy_result = import_asset_data(
        map_data,
        relative_map,
        bpy.context,
        ImportSettings(
            import_textures=False,
            create_armatures=False,
            max_map_instances=1,
            map_placement_mode="COLLECTION_INSTANCES",
        ),
        source=DirectorySource(game_root),
        virtual_path=relative_map,
    )
    assert legacy_result.instances == 1
    legacy_placements = [
        obj
        for obj in bpy.data.objects
        if obj.get("chromehounds_map_placement", False)
        and obj.get("chromehounds_placement_mode") == "COLLECTION_INSTANCES"
    ]
    assert len(legacy_placements) == 1
    assert legacy_placements[0].instance_type == "COLLECTION"
    assert legacy_placements[0].instance_collection is not None
    build_manifest = {
        "version": 1,
        "name": "Smoke Build",
        "parts": [
            {"id": "base", "asset": "parts/CK_CA001.bnd"},
            {
                "id": "weapon",
                "asset": "parts/WP_GN001.bnd",
                "parent": "base",
                "socket": 200,
                "location": [1.0, 2.0, 3.0],
            },
        ],
    }
    build_result = import_asset_data(
        json.dumps(build_manifest).encode("utf-8"),
        "smoke.chbuild.json",
        bpy.context,
        ImportSettings(import_textures=True, import_animations=False),
        source=DirectorySource(game_root),
        virtual_path="smoke.chbuild.json",
    )
    assert build_result.models == 2
    assert build_result.images == 6
    weapon = next(
        obj
        for obj in bpy.data.objects
        if obj.get("chromehounds_build_part_id") == "weapon"
    )
    assert weapon.parent is not None
    assert weapon.parent.get("chromehounds_reference_id") == 200
    assert tuple(round(value, 4) for value in weapon.matrix_basis.translation) == (
        1.0,
        2.0,
        3.0,
    )
    print(
        "Chromehounds Blender smoke test passed:",
        part_result,
        character_result,
        map_result,
        build_result,
    )
    with tempfile.TemporaryDirectory(prefix="chromehounds-blender-smoke-") as temp:
        saved_path = Path(temp) / "texture-persistence.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(saved_path), check_existing=False)
        bpy.ops.wm.open_mainfile(filepath=str(saved_path), load_ui=False)
        for image_name in ("AX_AM001", "AX_AM001_b", "AX_AM001_c"):
            image = bpy.data.images[image_name]
            assert image.use_fake_user
            assert image.packed_file is not None
            pixels = np.empty(len(image.pixels), dtype=np.float32)
            image.pixels.foreach_get(pixels)
            rgb = pixels.reshape(-1, image.channels)[:, :3]
            assert float(np.ptp(rgb)) > 0.01, image_name


if __name__ == "__main__":
    main()
