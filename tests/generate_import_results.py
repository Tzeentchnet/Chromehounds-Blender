from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


CASES = {
    "hound_iso": ("01_hound_part_from_iso", "AX_AM001_from_ISO"),
    "animated_character": ("02_animated_character", "c0101_animated"),
    "prop": ("03_textured_prop", "o06-030_textured"),
    "level": ("04_full_level", "m01_001_full_level"),
    "assembly": ("05_socket_assembly", "hound_socket_assembly"),
}


def parse_args() -> argparse.Namespace:
    try:
        separator = sys.argv.index("--")
    except ValueError:
        script_args: list[str] = []
    else:
        script_args = sys.argv[separator + 1 :]
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--game-root", type=Path, required=True)
    parser.add_argument("--iso", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args(script_args)


def clear_factory_scene() -> None:
    for obj in tuple(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in tuple(bpy.data.collections):
        bpy.data.collections.remove(collection)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.armatures,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.materials,
        bpy.data.actions,
    ):
        for datablock in tuple(datablocks):
            datablocks.remove(datablock)
    for image in tuple(bpy.data.images):
        if image.name not in {"Render Result", "Viewer Node"}:
            bpy.data.images.remove(image)
    scene = bpy.context.scene
    scene.frame_start = 0
    scene.frame_end = 250
    scene.frame_set(0)


def invoke_import(args: argparse.Namespace) -> tuple[set[str], dict[str, object]]:
    scene = bpy.context.scene
    props = scene.chromehounds_import
    validation: dict[str, object] = {}

    if args.case == "hound_iso":
        props.source_path = str(args.iso)
        props.asset_path = "parts/AX_AM001.bnd"
        props.import_textures = True
        props.import_animations = True
        props.create_armatures = True
        result = bpy.ops.chromehounds.import_from_source()
        mesh = bpy.data.objects.get("AX_AM001_Mesh_000")
        assert mesh is not None and mesh.type == "MESH"
        assert len(mesh.data.vertices) == 748
        expected_images = {"AX_AM001", "AX_AM001_b", "AX_AM001_c"}
        assert expected_images.issubset(bpy.data.images.keys())
        validation.update(
            vertices=len(mesh.data.vertices),
            polygons=len(mesh.data.polygons),
            texture_images=sorted(expected_images),
            source="retail ISO",
        )
    elif args.case == "animated_character":
        result = bpy.ops.chromehounds.import_asset(
            filepath=str(args.game_root / "chr" / "c0101.bnd"),
            import_textures=True,
            import_animations=True,
            create_armatures=True,
        )
        armature = bpy.data.objects.get("c0101")
        action = bpy.data.actions.get("c0101_Clip_011")
        assert armature is not None and armature.type == "ARMATURE"
        assert len(armature.data.bones) == 23
        assert action is not None
        armature.animation_data_create().action = action
        scene.frame_start = int(action.frame_range[0])
        scene.frame_end = int(action.frame_range[1])
        scene.frame_set(30)
        assert len(bpy.data.actions) == 4
        validation.update(
            bones=len(armature.data.bones),
            actions=sorted(action.name for action in bpy.data.actions),
            displayed_action=action.name,
            displayed_frame=scene.frame_current,
        )
    elif args.case == "prop":
        prop_path = (
            args.game_root
            / "mapdata"
            / "obj"
            / "o06-030"
            / "o06-030.bnd"
        )
        result = bpy.ops.chromehounds.import_asset(
            filepath=str(prop_path),
            import_textures=True,
            import_animations=False,
            create_armatures=False,
        )
        meshes = [
            obj
            for obj in bpy.data.objects
            if obj.type == "MESH"
            and not obj.get("chromehounds_test_presentation", False)
        ]
        imported_images = [
            image
            for image in bpy.data.images
            if image.name not in {"Render Result", "Viewer Node"}
        ]
        assert meshes
        assert len(imported_images) >= 2
        validation.update(
            mesh_objects=len(meshes),
            texture_images=sorted(image.name for image in imported_images),
            compact_xpr_descriptor_covered=True,
        )
    elif args.case == "level":
        level_path = (
            args.game_root
            / "mapdata"
            / "map"
            / "m01_001"
            / "m01_001.msb"
        )
        result = bpy.ops.chromehounds.import_asset(
            filepath=str(level_path),
            import_textures=True,
            import_animations=False,
            import_external_assets=True,
            create_armatures=False,
            map_placement_mode="LINKED_OBJECTS",
            max_map_instances=0,
        )
        placement_roots = [
            obj
            for obj in bpy.data.objects
            if obj.get("chromehounds_map_placement", False)
            and obj.get("chromehounds_placement_mode") == "LINKED_OBJECTS"
        ]
        assert len(placement_roots) == 8378
        assert not any(
            obj.instance_type == "COLLECTION" for obj in placement_roots
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
        presentation_hidden = []
        for root in placement_roots:
            if root.get("chromehounds_model_name") == "o40-025":
                hierarchy = [root]
                for obj in hierarchy:
                    hierarchy.extend(obj.children)
                for obj in hierarchy:
                    obj.hide_render = True
                presentation_hidden.append(root.name)
        validation.update(
            placements=len(placement_roots),
            complete_level=True,
            expected_placements=8378,
            verified_opaque_material=opaque_material.name,
            presentation_hidden_environment_placements=presentation_hidden,
        )
    else:
        props.source_path = str(args.game_root)
        result = bpy.ops.chromehounds.import_asset(
            filepath=str(args.manifest),
            import_textures=True,
            import_animations=False,
            create_armatures=True,
        )
        part_roots = [
            obj
            for obj in bpy.data.objects
            if obj.get("chromehounds_build_part_id") is not None
        ]
        weapon = next(
            (
                obj
                for obj in part_roots
                if obj.get("chromehounds_build_part_id") == "weapon"
            ),
            None,
        )
        assert len(part_roots) == 2
        assert weapon is not None and weapon.parent is not None
        assert weapon.parent.get("chromehounds_reference_id") == 200
        validation.update(
            parts=sorted(
                str(obj.get("chromehounds_build_part_id")) for obj in part_roots
            ),
            weapon_parent=weapon.parent.name,
            socket_reference=weapon.parent.get("chromehounds_reference_id"),
        )

    assert result == {"FINISHED"}, result
    texture_images = [
        image
        for image in bpy.data.images
        if image.get("chromehounds_format") is not None
    ]
    assert texture_images
    assert all(image.use_fake_user for image in texture_images)
    has_texture_detail = False
    for image in texture_images:
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        rgb = pixels.reshape(-1, image.channels)[:, :3]
        if float(np.ptp(rgb)) > 0.01:
            has_texture_detail = True
            break
    assert has_texture_detail
    validation["embedded_texture_images"] = len(texture_images)
    return result, validation


def object_bounds() -> tuple[Vector, Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points: list[Vector] = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.users_scene:
            continue
        if obj.get("chromehounds_test_presentation", False):
            continue
        evaluated = obj.evaluated_get(depsgraph)
        points.extend(
            evaluated.matrix_world @ Vector(corner)
            for corner in evaluated.bound_box
        )
    if not points:
        points = [
            obj.matrix_world.translation.copy()
            for obj in bpy.data.objects
            if obj.instance_type == "COLLECTION" and obj.users_scene
        ]
    if not points:
        raise RuntimeError("Imported scene has no visible geometry or instances")
    minimum = Vector(
        tuple(min(point[axis] for point in points) for axis in range(3))
    )
    maximum = Vector(
        tuple(max(point[axis] for point in points) for axis in range(3))
    )
    return minimum, maximum


def level_bounds() -> tuple[Vector, Vector]:
    points = [
        obj.matrix_world.translation.copy()
        for obj in bpy.data.objects
        if obj.get("chromehounds_map_placement", False) and obj.users_scene
    ]
    if not points:
        return object_bounds()
    minimum = Vector(
        tuple(min(point[axis] for point in points) for axis in range(3))
    )
    maximum = Vector(
        tuple(max(point[axis] for point in points) for axis in range(3))
    )
    horizontal = max(maximum.x - minimum.x, maximum.y - minimum.y, 1.0)
    minimum.z -= horizontal * 0.03
    maximum.z += horizontal * 0.12
    return minimum, maximum


def create_camera(
    minimum: Vector,
    maximum: Vector,
    *,
    level: bool,
) -> bpy.types.Object:
    center = (minimum + maximum) * 0.5
    dimensions = maximum - minimum
    extent = max(dimensions.x, dimensions.y, dimensions.z, 1.0)
    direction = (
        Vector((1.15, -1.35, 1.25))
        if level
        else Vector((1.35, -1.65, 1.05))
    ).normalized()
    camera_data = bpy.data.cameras.new("Test Camera")
    camera = bpy.data.objects.new("Test Camera", camera_data)
    camera["chromehounds_test_presentation"] = True
    bpy.context.scene.collection.objects.link(camera)
    camera.location = center + direction * extent * 2.5
    camera.rotation_euler = (center - camera.location).to_track_quat(
        "-Z",
        "Y",
    ).to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = extent * (1.9 if level else 1.68)
    camera_data.clip_start = max(0.001, extent / 100_000.0)
    camera_data.clip_end = max(10_000.0, extent * 10.0)
    bpy.context.scene.camera = camera
    return camera


def add_area_light(
    name: str,
    location: Vector,
    target: Vector,
    energy: float,
    size: float,
) -> None:
    light_data = bpy.data.lights.new(name, "AREA")
    light_data.energy = energy
    light_data.shape = "DISK"
    light_data.size = size
    light = bpy.data.objects.new(name, light_data)
    light["chromehounds_test_presentation"] = True
    light.location = location
    light.rotation_euler = (target - location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.collection.objects.link(light)


def setup_eevee(minimum: Vector, maximum: Vector) -> None:
    scene = bpy.context.scene
    center = (minimum + maximum) * 0.5
    dimensions = maximum - minimum
    extent = max(dimensions.x, dimensions.y, dimensions.z, 1.0)
    create_camera(minimum, maximum, level=False)

    floor_z = minimum.z - extent * 0.015
    bpy.ops.mesh.primitive_plane_add(
        size=extent * 5.0,
        location=(center.x, center.y, floor_z),
    )
    floor = bpy.context.object
    floor.name = "Test Floor"
    floor["chromehounds_test_presentation"] = True
    floor_material = bpy.data.materials.new("Test Floor Material")
    floor_material.diffuse_color = (0.035, 0.045, 0.06, 1.0)
    floor_material.use_nodes = True
    floor_bsdf = floor_material.node_tree.nodes.get("Principled BSDF")
    if floor_bsdf is not None:
        floor_bsdf.inputs["Base Color"].default_value = (
            0.035,
            0.045,
            0.06,
            1.0,
        )
        floor_bsdf.inputs["Roughness"].default_value = 0.72
    floor.data.materials.append(floor_material)

    intensity = 2_800.0 * max(1.0, (extent / 3.0) ** 2)
    add_area_light(
        "Key Light",
        center + Vector((extent * 1.3, -extent * 1.4, extent * 1.8)),
        center,
        intensity,
        extent * 0.9,
    )
    add_area_light(
        "Fill Light",
        center + Vector((-extent * 1.5, -extent * 0.4, extent * 0.9)),
        center,
        intensity * 0.45,
        extent * 1.1,
    )
    add_area_light(
        "Rim Light",
        center + Vector((extent * 0.1, extent * 1.6, extent * 1.4)),
        center,
        intensity * 0.65,
        extent * 0.8,
    )

    scene.render.engine = "BLENDER_EEVEE"
    world = scene.world or bpy.data.worlds.new("Test World")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = (0.018, 0.025, 0.04, 1.0)
        background.inputs["Strength"].default_value = 0.65


def setup_workbench(minimum: Vector, maximum: Vector) -> None:
    scene = bpy.context.scene
    create_camera(minimum, maximum, level=True)
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "FLAT"
    shading.color_type = "TEXTURE"
    shading.show_shadows = False
    shading.show_cavity = True
    shading.cavity_type = "WORLD"
    shading.curvature_ridge_factor = 1.8
    shading.curvature_valley_factor = 1.2
    shading.background_type = "VIEWPORT"
    shading.background_color = (0.018, 0.025, 0.04)


def configure_render(path: Path) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.filepath = str(path)
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0.8


def collect_metrics() -> dict[str, object]:
    imported_meshes = [
        mesh
        for mesh in bpy.data.meshes
        if not mesh.name.startswith("Test Floor")
    ]
    imported_images = [
        image
        for image in bpy.data.images
        if image.name not in {"Render Result", "Viewer Node"}
    ]
    return {
        "objects": len(
            [
                obj
                for obj in bpy.data.objects
                if not obj.get("chromehounds_test_presentation", False)
            ]
        ),
        "collections": len(bpy.data.collections),
        "mesh_datablocks": len(imported_meshes),
        "vertices": sum(len(mesh.vertices) for mesh in imported_meshes),
        "polygons": sum(len(mesh.polygons) for mesh in imported_meshes),
        "materials": len(
            [
                material
                for material in bpy.data.materials
                if not material.name.startswith("Test Floor")
            ]
        ),
        "images": len(imported_images),
        "image_names": sorted(image.name for image in imported_images),
        "armatures": len(bpy.data.armatures),
        "actions": len(bpy.data.actions),
        "collection_instances": sum(
            obj.instance_type == "COLLECTION" for obj in bpy.data.objects
        ),
        "map_placements": sum(
            obj.get("chromehounds_map_placement", False)
            for obj in bpy.data.objects
        ),
    }


def main() -> None:
    args = parse_args()
    case_folder, stem = CASES[args.case]
    output_dir = args.output_root / case_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = output_dir / f"{stem}.png"
    blend_path = output_dir / f"{stem}.blend"
    report_path = output_dir / "report.json"

    if not hasattr(bpy.context.scene, "chromehounds_import"):
        raise RuntimeError("The packaged Chromehounds extension is not enabled")
    clear_factory_scene()
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    result, validation = invoke_import(args)
    import_seconds = time.perf_counter() - started
    metrics = collect_metrics()

    if args.case == "level":
        minimum, maximum = level_bounds()
        setup_workbench(minimum, maximum)
    else:
        minimum, maximum = object_bounds()
        setup_eevee(minimum, maximum)
    configure_render(screenshot_path)
    bpy.context.scene["chromehounds_test_case"] = args.case
    bpy.context.scene["chromehounds_import_seconds"] = import_seconds
    bpy.context.scene["chromehounds_validation"] = json.dumps(validation)

    render_started = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    render_seconds = time.perf_counter() - render_started
    if not screenshot_path.is_file() or screenshot_path.stat().st_size == 0:
        raise RuntimeError(f"Screenshot was not written: {screenshot_path}")

    bpy.context.preferences.filepaths.use_file_compression = True
    bpy.ops.wm.save_as_mainfile(
        filepath=str(blend_path),
        check_existing=False,
    )
    if not blend_path.is_file() or blend_path.stat().st_size == 0:
        raise RuntimeError(f"Blend file was not written: {blend_path}")

    report = {
        "case": args.case,
        "status": "passed",
        "operator_result": sorted(result),
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "blender_version": bpy.app.version_string,
        "import_seconds": round(import_seconds, 3),
        "render_seconds": round(render_seconds, 3),
        "bounds": {
            "minimum": list(minimum),
            "maximum": list(maximum),
        },
        "validation": validation,
        "metrics": metrics,
        "files": {
            "blend": str(blend_path),
            "blend_bytes": blend_path.stat().st_size,
            "screenshot": str(screenshot_path),
            "screenshot_bytes": screenshot_path.stat().st_size,
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("CHROMEHOUNDS_IMPORT_RESULT", json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
