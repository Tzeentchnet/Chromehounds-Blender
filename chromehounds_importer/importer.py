from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import bpy
from mathutils import Euler, Matrix, Vector

from .blender_data import (
    BuiltModel,
    TextureLibrary,
    build_actions,
    build_model,
    map_transform,
)
from .bnd import BndArchive, BndEntry, parse_bnd
from .errors import AssetNotFoundError, FormatError
from .m2a import parse_m2a
from .model import Mdl4, parse_mdl4, parse_smd4
from .msb import MsbMap, parse_msb
from .source import (
    AssetSource,
    DirectorySource,
    infer_game_root,
    normalise_asset_path,
    open_asset_source,
)
from .texture import parse_xpr

_LOD_PATTERN = re.compile(r"_L\d\d$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ImportSettings:
    unit_scale: float = 0.01
    animation_fps: int = 30
    import_lods: bool = False
    import_shadows: bool = False
    import_animations: bool = True
    import_textures: bool = True
    import_external_assets: bool = True
    create_armatures: bool = True
    max_map_instances: int = 0
    map_placement_mode: str = "LINKED_OBJECTS"


@dataclass(frozen=True, slots=True)
class ImportResult:
    collection: bpy.types.Collection | None
    models: int = 0
    meshes: int = 0
    materials: int = 0
    images: int = 0
    actions: int = 0
    instances: int = 0
    warnings: tuple[str, ...] = ()


def _new_top_collection(name: str, context: bpy.types.Context) -> bpy.types.Collection:
    collection = bpy.data.collections.new(name)
    context.collection.children.link(collection)
    return collection


def _bundle_texture_library(
    archive: BndArchive,
    *,
    enabled: bool,
    existing: TextureLibrary | None = None,
) -> TextureLibrary:
    library = existing or TextureLibrary()
    if not enabled:
        return library
    for entry in archive.entries:
        if entry.data[:4] != b"XPR2":
            continue
        package = parse_xpr(entry.data, entry.name)
        for texture in package.textures:
            library.add(texture)
    return library


def _is_lod(stem: str) -> bool:
    return _LOD_PATTERN.search(stem) is not None


def _model_entries(archive: BndArchive, include_lods: bool) -> tuple[BndEntry, ...]:
    return tuple(
        entry
        for entry in archive.entries
        if entry.data[:4] == b"MDL4"
        and (include_lods or not _is_lod(PurePosixPath(entry.name).stem))
    )


def import_bnd(
    data: bytes,
    label: str,
    context: bpy.types.Context,
    settings: ImportSettings,
    *,
    parent_collection: bpy.types.Collection | None = None,
    link_parent: bool = True,
    texture_library: TextureLibrary | None = None,
    only_model: str | None = None,
) -> tuple[ImportResult, dict[str, tuple[Mdl4, BuiltModel]], TextureLibrary]:
    archive = parse_bnd(data, label)
    bundle_name = Path(label).stem or "Chromehounds_Bundle"
    if parent_collection is None:
        parent_collection = bpy.data.collections.new(bundle_name)
        if link_parent:
            context.collection.children.link(parent_collection)
    textures = _bundle_texture_library(
        archive,
        enabled=settings.import_textures,
        existing=texture_library,
    )
    entries = _model_entries(archive, settings.import_lods)
    if only_model:
        wanted = only_model.casefold()
        exact = [
            entry
            for entry in entries
            if PurePosixPath(entry.name).stem.casefold() == wanted
        ]
        if exact:
            entries = (exact[0],)
        else:
            entries = tuple(
                entry
                for entry in entries
                if not _is_lod(PurePosixPath(entry.name).stem)
            )[:1]
    if not entries:
        raise FormatError(f"{label}: BND contains no importable MDL4 model")

    built_by_stem: dict[str, tuple[Mdl4, BuiltModel]] = {}
    mesh_count = 0
    material_count = 0
    for model_index, entry in enumerate(entries):
        stem = PurePosixPath(entry.name).stem
        model = parse_mdl4(entry.data, f"{label}:{entry.name}")
        built = build_model(
            model,
            stem,
            parent_collection,
            textures,
            unit_scale=settings.unit_scale,
            create_armature=settings.create_armatures,
            material_prefix=f"{bundle_name}_" if len(entries) > 1 else "",
        )
        if _is_lod(stem) and model_index > 0:
            built.collection.hide_viewport = True
            built.collection.hide_render = True
        built_by_stem[stem.casefold()] = (model, built)
        mesh_count += len(built.mesh_objects)
        material_count += len(model.materials)

    actions = 0
    warnings: list[str] = []
    if settings.import_animations:
        for entry in archive.entries:
            if entry.data[:4] != b"M2Af":
                continue
            stem = PurePosixPath(entry.name).stem.casefold()
            target = built_by_stem.get(stem)
            if target is None and len(built_by_stem) == 1:
                target = next(iter(built_by_stem.values()))
            if target is None:
                warnings.append(
                    f"Animation {entry.name} has no matching model in {label}"
                )
                continue
            model, built = target
            animation = parse_m2a(entry.data, f"{label}:{entry.name}")
            actions += len(
                build_actions(
                    animation,
                    model,
                    built,
                    unit_scale=settings.unit_scale,
                    name_prefix=f"{PurePosixPath(entry.name).stem}_",
                )
            )
        if actions:
            context.scene.render.fps = settings.animation_fps

    if settings.import_shadows:
        shadow_parent = bpy.data.collections.new(f"{bundle_name}_Shadows")
        parent_collection.children.link(shadow_parent)
        shadow_parent.hide_render = True
        for entry in archive.entries:
            if entry.data[:4] != b"SMD4":
                continue
            stem = PurePosixPath(entry.name).stem
            if not settings.import_lods and _is_lod(stem):
                continue
            shadow = parse_smd4(entry.data, f"{label}:{entry.name}")
            built = build_model(
                shadow,
                f"{stem}_Shadow",
                shadow_parent,
                textures,
                unit_scale=settings.unit_scale,
                create_armature=False,
            )
            for obj in built.mesh_objects:
                obj.display_type = "WIRE"
            mesh_count += len(built.mesh_objects)

    return (
        ImportResult(
            collection=parent_collection,
            models=len(built_by_stem),
            meshes=mesh_count,
            materials=material_count,
            images=textures.image_count,
            actions=actions,
            warnings=tuple(warnings),
        ),
        built_by_stem,
        textures,
    )


def _adjacent_texture_library(path: Path, enabled: bool) -> TextureLibrary:
    library = TextureLibrary()
    if not enabled:
        return library
    for candidate in sorted(path.parent.glob("*.xp[rc]")):
        package = parse_xpr(candidate.read_bytes(), str(candidate))
        for texture in package.textures:
            library.add(texture)
    return library


def _source_texture_library(
    source: AssetSource,
    virtual_path: str,
    enabled: bool,
) -> TextureLibrary:
    library = TextureLibrary()
    if not enabled:
        return library
    parent = PurePosixPath(virtual_path).parent
    prefix = "" if str(parent) == "." else f"{parent.as_posix().casefold()}/"
    candidates = [
        path
        for path in source.paths()
        if PurePosixPath(path).suffix.casefold() in {".xpr", ".xpc"}
        and path.casefold().startswith(prefix)
        and "/" not in path[len(prefix) :]
    ]
    if virtual_path.casefold().startswith("parts/") and source.exists(
        "parts/parts_common.xpr"
    ):
        candidates.append("parts/parts_common.xpr")
    for candidate in dict.fromkeys(candidates):
        package = parse_xpr(source.read(candidate), candidate)
        for texture in package.textures:
            library.add(texture)
    return library


def import_standalone_model(
    path: Path,
    context: bpy.types.Context,
    settings: ImportSettings,
    *,
    textures: TextureLibrary | None = None,
) -> ImportResult:
    data = path.read_bytes()
    textures = textures or _adjacent_texture_library(path, settings.import_textures)
    top = _new_top_collection(path.stem, context)
    if data[:4] == b"MDL4":
        model = parse_mdl4(data, str(path))
        built = build_model(
            model,
            path.stem,
            top,
            textures,
            unit_scale=settings.unit_scale,
            create_armature=settings.create_armatures,
        )
        return ImportResult(
            collection=top,
            models=1,
            meshes=len(built.mesh_objects),
            materials=len(model.materials),
            images=textures.image_count,
        )
    if data[:4] == b"SMD4":
        model = parse_smd4(data, str(path))
        built = build_model(
            model,
            path.stem,
            top,
            textures,
            unit_scale=settings.unit_scale,
            create_armature=False,
        )
        return ImportResult(
            collection=top,
            models=1,
            meshes=len(built.mesh_objects),
        )
    raise FormatError(f"{path}: expected MDL4 or SMD4 data")


def import_xpr(
    data: bytes,
    label: str,
    settings: ImportSettings,
) -> ImportResult:
    package = parse_xpr(data, label)
    library = TextureLibrary(package.textures)
    images = library.import_all()
    return ImportResult(collection=None, images=len(images))


def import_standalone_animation(
    data: bytes,
    label: str,
    context: bpy.types.Context,
    settings: ImportSettings,
    *,
    source: AssetSource | None,
    virtual_path: str | None,
    local_path: Path | None,
) -> ImportResult:
    model_data, model_label = _read_companion(
        source,
        virtual_path,
        local_path,
        ".mdl",
    )
    model = parse_mdl4(model_data, model_label)
    animation = parse_m2a(data, label)
    top = _new_top_collection(PurePosixPath(label).stem, context)
    built = build_model(
        model,
        PurePosixPath(model_label).stem,
        top,
        TextureLibrary(),
        unit_scale=settings.unit_scale,
        create_armature=True,
    )
    actions = build_actions(
        animation,
        model,
        built,
        unit_scale=settings.unit_scale,
        name_prefix=f"{PurePosixPath(label).stem}_",
    )
    context.scene.render.fps = settings.animation_fps
    return ImportResult(
        collection=top,
        models=1,
        meshes=len(built.mesh_objects),
        materials=len(model.materials),
        actions=len(actions),
    )


def _read_companion(
    source: AssetSource | None,
    virtual_path: str | None,
    local_path: Path | None,
    suffix: str,
) -> tuple[bytes, str]:
    if source is not None and virtual_path is not None:
        path = str(PurePosixPath(virtual_path).with_suffix(suffix))
        if not source.exists(path):
            raise AssetNotFoundError(f"Required companion asset {path!r} does not exist")
        return source.read(path), path
    if local_path is None:
        raise AssetNotFoundError(f"Cannot resolve required {suffix} companion")
    path = local_path.with_suffix(suffix)
    if not path.is_file():
        raise AssetNotFoundError(f"Required companion asset {path} does not exist")
    return path.read_bytes(), str(path)


def _read_source_xpr(
    source: AssetSource,
    path: str,
    library: TextureLibrary,
) -> None:
    if not source.exists(path):
        return
    package = parse_xpr(source.read(path), path)
    for texture in package.textures:
        library.add(texture)


def _external_bundle_path(source: AssetSource, model_name: str, model_type: int) -> str:
    if model_type == 1:
        direct = f"mapdata/obj/{model_name}/{model_name}.bnd"
        prefix = "mapdata/obj"
    elif model_type == 2:
        direct = f"chr/{model_name}.bnd"
        prefix = "chr"
    else:
        raise AssetNotFoundError(
            f"MSB model {model_name!r} does not use an external bundle"
        )
    if source.exists(direct):
        return direct
    matches = source.find_by_basename(
        f"{model_name}.bnd",
        preferred_prefix=prefix,
    )
    if not matches:
        raise AssetNotFoundError(
            f"Could not locate the external bundle for MSB model {model_name!r}"
        )
    return matches[0]


def _linked_model_placement(
    built: BuiltModel,
    name: str,
    matrix: Matrix,
    collection: bpy.types.Collection,
) -> bpy.types.Object:
    source_objects = tuple(built.collection.objects)
    copies: dict[bpy.types.Object, bpy.types.Object] = {}
    mesh_indices = {
        mesh_object: index
        for index, mesh_object in enumerate(built.mesh_objects)
    }
    for source_object in source_objects:
        duplicate = source_object.copy()
        duplicate.parent = None
        if source_object == built.root:
            duplicate.name = name
        elif source_object in mesh_indices:
            duplicate.name = f"{name}_Mesh_{mesh_indices[source_object]:03d}"
        else:
            duplicate.name = f"{name}_{source_object.name}"
        duplicate["chromehounds_placement_name"] = name
        duplicate["chromehounds_source_model"] = built.collection.name
        collection.objects.link(duplicate)
        copies[source_object] = duplicate

    for source_object, duplicate in copies.items():
        source_parent = source_object.parent
        if source_parent in copies:
            duplicate.parent = copies[source_parent]
            duplicate.parent_type = source_object.parent_type
            duplicate.parent_bone = source_object.parent_bone
            duplicate.matrix_parent_inverse = source_object.matrix_parent_inverse.copy()
            duplicate.matrix_basis = source_object.matrix_basis.copy()

    root = copies[built.root]
    root.matrix_world = matrix @ built.root.matrix_world
    return root


def import_msb(
    data: bytes,
    label: str,
    context: bpy.types.Context,
    settings: ImportSettings,
    *,
    source: AssetSource | None = None,
    virtual_path: str | None = None,
    local_path: Path | None = None,
) -> ImportResult:
    msb = parse_msb(data, label)
    bundle_data, bundle_label = _read_companion(
        source,
        virtual_path,
        local_path,
        ".bnd",
    )
    map_bundle = parse_bnd(bundle_data, bundle_label)
    map_name = PurePosixPath(virtual_path).stem if virtual_path else local_path.stem
    textures = _bundle_texture_library(
        map_bundle,
        enabled=settings.import_textures,
    )
    if source is not None and settings.import_textures:
        _read_source_xpr(source, f"mapdata/tex/{map_name[:3]}.xpr", textures)

    top = _new_top_collection(map_name, context)
    placements = {
        0: bpy.data.collections.new("Map Pieces"),
        1: bpy.data.collections.new("Objects"),
        2: bpy.data.collections.new("Characters"),
    }
    for collection in placements.values():
        top.children.link(collection)
    source_root = bpy.data.collections.new(f"{map_name}_SourceModels")

    map_entries = {
        PurePosixPath(entry.name).stem.casefold(): entry
        for entry in _model_entries(map_bundle, include_lods=False)
    }
    used_model_indices = {
        part.model_index
        for part in msb.parts[: settings.max_map_instances or None]
        if 0 <= part.model_index < len(msb.models)
    }
    built_models: dict[int, BuiltModel] = {}
    warnings: list[str] = []
    mesh_count = 0
    material_count = 0
    for model_index in sorted(used_model_indices):
        definition = msb.models[model_index]
        try:
            if definition.model_type == 0:
                entry = map_entries.get(definition.name.casefold())
                if entry is None:
                    raise AssetNotFoundError(
                        f"{bundle_label} has no model named {definition.name}.mdl"
                    )
                model = parse_mdl4(entry.data, f"{bundle_label}:{entry.name}")
            else:
                if not settings.import_external_assets:
                    continue
                if source is None:
                    raise AssetNotFoundError(
                        "External map objects require an extracted-game or ISO source"
                    )
                external_path = _external_bundle_path(
                    source,
                    definition.name,
                    definition.model_type,
                )
                external_archive = parse_bnd(
                    source.read(external_path),
                    external_path,
                )
                for texture in _source_texture_library(
                    source,
                    external_path,
                    settings.import_textures,
                ):
                    textures.add(texture)
                _bundle_texture_library(
                    external_archive,
                    enabled=settings.import_textures,
                    existing=textures,
                )
                candidates = _model_entries(external_archive, include_lods=False)
                exact = [
                    entry
                    for entry in candidates
                    if PurePosixPath(entry.name).stem.casefold()
                    == definition.name.casefold()
                ]
                if not candidates:
                    raise FormatError(f"{external_path} contains no MDL4 model")
                entry = exact[0] if exact else candidates[0]
                model = parse_mdl4(entry.data, f"{external_path}:{entry.name}")
            built = build_model(
                model,
                definition.name,
                source_root,
                textures,
                unit_scale=settings.unit_scale,
                create_armature=False,
                material_prefix=f"{map_name}_",
                double_sided=True,
            )
            built_models[model_index] = built
            mesh_count += len(built.mesh_objects)
            material_count += len(model.materials)
        except (AssetNotFoundError, FormatError, OSError) as exc:
            warnings.append(str(exc))

    placement_mode = settings.map_placement_mode
    if placement_mode not in {"LINKED_OBJECTS", "COLLECTION_INSTANCES"}:
        raise FormatError(f"Unsupported map placement mode {placement_mode!r}")
    if placement_mode == "LINKED_OBJECTS":
        for model_index, built in built_models.items():
            built.root.name = f"__CH_SOURCE_{model_index:04d}_Root"
            for mesh_index, mesh_object in enumerate(built.mesh_objects):
                mesh_object.name = (
                    f"__CH_SOURCE_{model_index:04d}_Mesh_{mesh_index:03d}"
                )

    placement_count = 0
    for part in msb.parts:
        if settings.max_map_instances and placement_count >= settings.max_map_instances:
            break
        built = built_models.get(part.model_index)
        if built is None:
            continue
        definition = msb.models[part.model_index]
        placement_matrix = map_transform(
            part.position,
            part.rotation,
            part.scale,
            settings.unit_scale,
        )
        if placement_mode == "LINKED_OBJECTS":
            placement = _linked_model_placement(
                built,
                part.name,
                placement_matrix,
                placements[definition.model_type],
            )
        else:
            placement = bpy.data.objects.new(part.name, None)
            placement.instance_type = "COLLECTION"
            placement.instance_collection = built.collection
            placement.matrix_world = placement_matrix
            placements[definition.model_type].objects.link(placement)
        placement["chromehounds_map_placement"] = True
        placement["chromehounds_placement_name"] = part.name
        placement["chromehounds_placement_mode"] = placement_mode
        placement["chromehounds_model_name"] = definition.name
        placement["chromehounds_model_index"] = part.model_index
        placement["chromehounds_part_type"] = part.part_type
        placement["chromehounds_part_id"] = part.part_id
        placement["chromehounds_layer_id"] = part.layer_id
        placement_count += 1

    if settings.max_map_instances and len(msb.parts) > settings.max_map_instances:
        warnings.append(
            f"Map import was intentionally limited to {settings.max_map_instances} "
            f"of {len(msb.parts)} placements"
        )
    return ImportResult(
        collection=top,
        models=len(built_models),
        meshes=mesh_count,
        materials=material_count,
        images=textures.image_count,
        instances=placement_count,
        warnings=tuple(warnings),
    )


def _manifest_matrix(part: dict) -> Matrix:
    raw_matrix = part.get("matrix")
    if raw_matrix is not None:
        if (
            not isinstance(raw_matrix, list)
            or len(raw_matrix) != 4
            or any(not isinstance(row, list) or len(row) != 4 for row in raw_matrix)
        ):
            raise FormatError("build part matrix must be a row-major 4x4 array")
        return Matrix(raw_matrix)
    location = part.get("location", [0.0, 0.0, 0.0])
    rotation = part.get("rotation_degrees", [0.0, 0.0, 0.0])
    scale = part.get("scale", [1.0, 1.0, 1.0])
    if not all(isinstance(value, list) and len(value) == 3 for value in (location, rotation, scale)):
        raise FormatError(
            "build part location, rotation_degrees, and scale must be three-value arrays"
        )
    return (
        Matrix.Translation(Vector(location))
        @ Euler(tuple(value * 0.017453292519943295 for value in rotation), "XYZ")
        .to_matrix()
        .to_4x4()
        @ Matrix.Diagonal((*scale, 1.0))
    )


def import_build_manifest(
    data: bytes,
    label: str,
    context: bpy.types.Context,
    settings: ImportSettings,
    source: AssetSource | None,
    *,
    local_path: Path | None = None,
) -> ImportResult:
    try:
        manifest = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormatError(f"{label}: invalid UTF-8 build manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise FormatError(f"{label}: build manifest must be an object with version 1")
    if source is None:
        source_value = manifest.get("source")
        if not isinstance(source_value, str) or not source_value:
            raise AssetNotFoundError(
                f"{label}: set a game source in Blender or add a source path to the manifest"
            )
        source_path = Path(source_value)
        if not source_path.is_absolute() and local_path is not None:
            source_path = local_path.parent / source_path
        source = open_asset_source(source_path)
    parts = manifest.get("parts")
    if not isinstance(parts, list) or not parts:
        raise FormatError(f"{label}: build manifest must contain a non-empty parts array")

    top = _new_top_collection(str(manifest.get("name") or Path(label).stem), context)
    built_parts: dict[str, BuiltModel] = {}
    mesh_count = material_count = image_count = action_count = 0
    warnings: list[str] = []
    for index, part in enumerate(parts):
        if not isinstance(part, dict) or not isinstance(part.get("asset"), str):
            raise FormatError(f"{label}: build part {index} requires an asset path")
        part_id = str(part.get("id") or f"part_{index:03d}")
        if part_id in built_parts:
            raise FormatError(f"{label}: duplicate build part ID {part_id!r}")
        model_name = part.get("model")
        if model_name is not None and not isinstance(model_name, str):
            raise FormatError(f"{label}: build part {part_id!r} model must be a string")
        asset_path = normalise_asset_path(part["asset"])
        part_collection = bpy.data.collections.new(part_id)
        top.children.link(part_collection)
        result, models, _textures = import_bnd(
            source.read(asset_path),
            asset_path,
            context,
            settings,
            parent_collection=part_collection,
            only_model=model_name,
        )
        model, built = next(iter(models.values()))
        built.root.matrix_world = _manifest_matrix(part)
        built.root["chromehounds_build_part_id"] = part_id
        built.root["chromehounds_asset"] = asset_path
        built_parts[part_id] = built
        mesh_count += result.meshes
        material_count += result.materials
        image_count += result.images
        action_count += result.actions
        warnings.extend(result.warnings)

    parents: dict[str, str] = {}
    for index, part in enumerate(parts):
        parent_id = part.get("parent")
        if parent_id is None:
            continue
        part_id = str(part.get("id") or f"part_{index:03d}")
        parents[part_id] = str(parent_id)
    for part_id in parents:
        seen: set[str] = set()
        current = part_id
        while current in parents:
            if current in seen:
                raise FormatError(
                    f"{label}: build part hierarchy contains a cycle at {current!r}"
                )
            seen.add(current)
            current = parents[current]

    for index, part in enumerate(parts):
        parent_id = part.get("parent")
        if parent_id is None:
            continue
        part_id = str(part.get("id") or f"part_{index:03d}")
        child = built_parts[part_id].root
        parent = built_parts.get(str(parent_id))
        if parent is None:
            raise FormatError(
                f"{label}: build part {part_id!r} references unknown parent {parent_id!r}"
            )
        socket = part.get("socket")
        target = parent.root
        if socket is not None:
            match = next(
                (
                    obj
                    for obj in parent.collection.all_objects
                    if obj.name == socket
                    or str(obj.get("chromehounds_reference_id", "")) == str(socket)
                ),
                None,
            )
            if match is None:
                raise FormatError(
                    f"{label}: parent {parent_id!r} has no socket {socket!r}"
                )
            target = match
        local = _manifest_matrix(part)
        child.parent = target
        child.matrix_parent_inverse.identity()
        child.matrix_basis = local

    return ImportResult(
        collection=top,
        models=len(built_parts),
        meshes=mesh_count,
        materials=material_count,
        images=image_count,
        actions=action_count,
        warnings=tuple(warnings),
    )


def import_asset_data(
    data: bytes,
    label: str,
    context: bpy.types.Context,
    settings: ImportSettings,
    *,
    source: AssetSource | None = None,
    virtual_path: str | None = None,
    local_path: Path | None = None,
) -> ImportResult:
    magic = data[:4]
    suffix = PurePosixPath(virtual_path or label).suffix.casefold()
    if magic == b"BND3":
        if source is not None and virtual_path is not None:
            textures = _source_texture_library(
                source,
                virtual_path,
                settings.import_textures,
            )
        elif local_path is not None:
            textures = _adjacent_texture_library(
                local_path,
                settings.import_textures,
            )
        else:
            textures = TextureLibrary()
        return import_bnd(
            data,
            label,
            context,
            settings,
            texture_library=textures,
        )[0]
    if magic in (b"MDL4", b"SMD4"):
        textures = (
            _source_texture_library(source, virtual_path, settings.import_textures)
            if source is not None and virtual_path is not None
            else TextureLibrary()
        )
        if local_path is None:
            top = _new_top_collection(PurePosixPath(label).stem, context)
            if magic == b"MDL4":
                model = parse_mdl4(data, label)
                built = build_model(
                    model,
                    PurePosixPath(label).stem,
                    top,
                    textures,
                    unit_scale=settings.unit_scale,
                    create_armature=settings.create_armatures,
                )
                return ImportResult(
                    collection=top,
                    models=1,
                    meshes=len(built.mesh_objects),
                    materials=len(model.materials),
                )
            model = parse_smd4(data, label)
            built = build_model(
                model,
                PurePosixPath(label).stem,
                top,
                textures,
                unit_scale=settings.unit_scale,
                create_armature=False,
            )
            return ImportResult(collection=top, models=1, meshes=len(built.mesh_objects))
        return import_standalone_model(
            local_path,
            context,
            settings,
            textures=textures if source is not None and virtual_path is not None else None,
        )
    if magic == b"XPR2":
        return import_xpr(data, label, settings)
    if magic == b"M2Af":
        return import_standalone_animation(
            data,
            label,
            context,
            settings,
            source=source,
            virtual_path=virtual_path,
            local_path=local_path,
        )
    if suffix == ".msb":
        return import_msb(
            data,
            label,
            context,
            settings,
            source=source,
            virtual_path=virtual_path,
            local_path=local_path,
        )
    if suffix == ".json" and label.casefold().endswith(".chbuild.json"):
        if source is None and local_path is not None:
            root = infer_game_root(local_path)
            if root is not None:
                source = DirectorySource(root)
        return import_build_manifest(
            data,
            label,
            context,
            settings,
            source,
            local_path=local_path,
        )
    raise FormatError(
        f"{label}: unsupported asset type or unrecognized magic {magic!r}"
    )
