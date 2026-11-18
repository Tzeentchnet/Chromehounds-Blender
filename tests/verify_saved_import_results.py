from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np


def parse_args() -> argparse.Namespace:
    separator = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args(sys.argv[separator + 1 :])


def verify_case(case: str) -> None:
    texture_images = [
        image
        for image in bpy.data.images
        if image.get("chromehounds_format") is not None
    ]
    assert texture_images
    assert all(image.use_fake_user for image in texture_images)
    assert all(image.packed_file is not None for image in texture_images)
    has_texture_detail = False
    for image in texture_images:
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        rgb = pixels.reshape(-1, image.channels)[:, :3]
        if float(np.ptp(rgb)) > 0.01:
            has_texture_detail = True
            break
    assert has_texture_detail

    if case == "hound_iso":
        mesh = bpy.data.objects.get("AX_AM001_Mesh_000")
        assert mesh is not None and len(mesh.data.vertices) == 748
        material = mesh.data.materials[0]
        assert not any(
            link.to_node.bl_idname == "ShaderNodeBsdfPrincipled"
            and link.to_socket.name == "Alpha"
            for link in material.node_tree.links
        )
        assert {"AX_AM001", "AX_AM001_b", "AX_AM001_c"}.issubset(
            bpy.data.images.keys()
        )
    elif case == "animated_character":
        armature = bpy.data.objects.get("c0101")
        assert armature is not None and len(armature.data.bones) == 23
        assert armature.animation_data is not None
        assert armature.animation_data.action is not None
        assert armature.animation_data.action.name == "c0101_Clip_011"
        assert bpy.context.scene.frame_current == 30
        assert len(bpy.data.actions) == 4
    elif case == "prop":
        assert {"o06-030", "o06-030_b"}.issubset(bpy.data.images.keys())
        assert sum(obj.type == "MESH" for obj in bpy.data.objects) >= 3
    elif case == "level":
        assert len(texture_images) > 0
        additive_materials = [
            material
            for material in bpy.data.materials
            if material.get("chromehounds_alpha_mapping") == "ADDITIVE"
        ]
        assert len(additive_materials) == 6
        for material in additive_materials:
            assert material.surface_render_method == "BLENDED"
            surface_link = next(
                link
                for link in material.node_tree.links
                if link.to_node.bl_idname == "ShaderNodeOutputMaterial"
                and link.to_socket.name == "Surface"
            )
            assert surface_link.from_node.bl_idname == "ShaderNodeAddShader"
            assert not any(
                link.to_node.bl_idname == "ShaderNodeBsdfPrincipled"
                and link.to_socket.name == "Alpha"
                for link in material.node_tree.links
            )
        placements = [
            obj
            for obj in bpy.data.objects
            if obj.get("chromehounds_map_placement", False)
            and obj.get("chromehounds_placement_mode") == "LINKED_OBJECTS"
        ]
        assert len(placements) == 8378
        assert not any(
            obj.instance_type == "COLLECTION" for obj in placements
        )
        opaque_material = bpy.data.materials["m01_001_艶無し.004"]
        assert opaque_material.get("chromehounds_alpha_mapping") == "OPAQUE"
        assert not opaque_material.use_backface_culling
        assert not opaque_material.use_transparency_overlap
        assert not opaque_material.use_transparent_shadow
        principled = next(
            node
            for node in opaque_material.node_tree.nodes
            if node.bl_idname == "ShaderNodeBsdfPrincipled"
        )
        assert principled.inputs["Alpha"].default_value == 1.0
        assert not principled.inputs["Alpha"].is_linked
        for placement_name, model_name in (
            ("o01-020_540", "o01-020"),
            ("m0015", "m0015"),
        ):
            root = next(
                obj
                for obj in placements
                if obj.get("chromehounds_placement_name") == placement_name
                and obj.get("chromehounds_model_name") == model_name
            )
            mesh_children = [
                child for child in root.children if child.type == "MESH"
            ]
            assert mesh_children
            assert all(mesh.users_scene for mesh in mesh_children)
            assert all(mesh.data.materials for mesh in mesh_children)
            assert all(
                material.get("chromehounds_alpha_mapping") == "OPAQUE"
                for mesh in mesh_children
                for material in mesh.data.materials
            )
            assert all(not mesh.show_transparent for mesh in mesh_children)
            bpy.ops.object.select_all(action="DESELECT")
            mesh_children[0].select_set(True)
            bpy.context.view_layer.objects.active = mesh_children[0]
            assert mesh_children[0].select_get()
            assert bpy.context.view_layer.objects.active == mesh_children[0]
        hidden_roots = [
            obj
            for obj in placements
            if obj.get("chromehounds_model_name") == "o40-025"
        ]
        assert len(hidden_roots) == 1
        hidden_hierarchy = [hidden_roots[0]]
        for current in hidden_hierarchy:
            hidden_hierarchy.extend(current.children)
        assert all(obj.hide_render for obj in hidden_hierarchy)
    elif case == "assembly":
        weapon = next(
            (
                obj
                for obj in bpy.data.objects
                if obj.get("chromehounds_build_part_id") == "weapon"
            ),
            None,
        )
        assert weapon is not None and weapon.parent is not None
        assert weapon.parent.get("chromehounds_reference_id") == 200
    else:
        raise AssertionError(f"Unknown saved test case {case!r}")


def main() -> None:
    args = parse_args()
    reports = sorted(args.output_root.glob("*/report.json"))
    assert len(reports) == 5, reports
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        blend_path = Path(report["files"]["blend"])
        screenshot_path = Path(report["files"]["screenshot"])
        assert blend_path.is_file() and blend_path.stat().st_size > 0
        assert screenshot_path.is_file() and screenshot_path.stat().st_size > 0
        bpy.ops.wm.open_mainfile(filepath=str(blend_path), load_ui=False)
        verify_case(report["case"])
        print(
            "SAVED_RESULT_OK",
            report["case"],
            blend_path,
            screenshot_path,
        )


if __name__ == "__main__":
    main()
