from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

from .errors import FormatError
from .m2a import M2aAnimation
from .model import Mdl4, ModelMaterial, ModelMesh, ModelNode, Smd4
from .texture import (
    FORMAT_CTX1,
    FORMAT_DXN,
    XprTexture,
    decode_texture,
)


@dataclass(slots=True)
class BuiltModel:
    collection: bpy.types.Collection
    root: bpy.types.Object
    armature: bpy.types.Object | None
    mesh_objects: tuple[bpy.types.Object, ...]
    actions: tuple[bpy.types.Action, ...]


def _source_local_matrix(
    translation: tuple[float, float, float],
    rotation: tuple[float, float, float],
    scale: tuple[float, float, float],
) -> Matrix:
    return (
        Matrix.Translation(translation)
        @ Euler(rotation, "XZY").to_matrix().to_4x4()
        @ Matrix.Diagonal((*scale, 1.0))
    )


_COORDINATE_MATRIX = Matrix(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
)


def _to_blender_matrix(source: Matrix, unit_scale: float) -> Matrix:
    converted = _COORDINATE_MATRIX @ source @ _COORDINATE_MATRIX.inverted()
    converted.translation *= unit_scale
    return converted


def _to_blender_points(values: np.ndarray, unit_scale: float) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float32)
    result[:, 0] = values[:, 0] * unit_scale
    result[:, 1] = -values[:, 2] * unit_scale
    result[:, 2] = values[:, 1] * unit_scale
    return result


def _to_blender_vectors(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float32)
    result[:, 0] = values[:, 0]
    result[:, 1] = -values[:, 2]
    result[:, 2] = values[:, 1]
    lengths = np.linalg.norm(result, axis=1)
    valid = lengths > 1e-8
    result[valid] /= lengths[valid, None]
    return result


def _world_matrices(nodes: tuple[ModelNode, ...]) -> tuple[Matrix, ...]:
    local = tuple(
        _source_local_matrix(node.translation, node.rotation, node.scale)
        for node in nodes
    )
    world: list[Matrix | None] = [None] * len(nodes)

    def resolve(index: int, active: set[int]) -> Matrix:
        existing = world[index]
        if existing is not None:
            return existing
        if index in active:
            raise FormatError(f"model node hierarchy contains a cycle at node {index}")
        active.add(index)
        parent = nodes[index].parent_index
        if parent < -1 or parent >= len(nodes):
            raise FormatError(
                f"model node {index} has invalid parent index {parent}"
            )
        result = local[index] if parent == -1 else resolve(parent, active) @ local[index]
        active.remove(index)
        world[index] = result
        return result

    return tuple(resolve(index, set()) for index in range(len(nodes)))


def _palette_bone(mesh: ModelMesh, palette_index: int, node_count: int) -> int:
    if not 0 <= palette_index < len(mesh.bone_palette):
        return -1
    node_index = int(mesh.bone_palette[palette_index])
    return node_index if 0 <= node_index < node_count else -1


def _vertex_bones(mesh: ModelMesh, node_count: int) -> np.ndarray:
    vertex_count = len(mesh.positions) if mesh.positions is not None else 0
    result = np.full(vertex_count, -1, dtype=np.int32)
    if mesh.normal_bone_indices is not None:
        for palette_index in np.unique(mesh.normal_bone_indices):
            node_index = _palette_bone(mesh, int(palette_index), node_count)
            if node_index >= 0:
                result[mesh.normal_bone_indices == palette_index] = node_index
    if mesh.bone_indices is not None:
        unresolved = result < 0
        for weight_slot in range(mesh.bone_indices.shape[1]):
            palette_indices = mesh.bone_indices[:, weight_slot]
            weighted = (
                mesh.bone_weights[:, weight_slot] > 1e-8
                if mesh.bone_weights is not None
                else np.ones(vertex_count, dtype=bool)
            )
            for palette_index in np.unique(palette_indices[unresolved & weighted]):
                node_index = _palette_bone(mesh, int(palette_index), node_count)
                if node_index >= 0:
                    selected = unresolved & weighted & (palette_indices == palette_index)
                    result[selected] = node_index
                    unresolved[selected] = False
    fallback = _palette_bone(mesh, mesh.default_bone_index, node_count)
    if fallback >= 0:
        result[result < 0] = fallback
    return result


def _bind_mesh(
    mesh: ModelMesh,
    nodes: tuple[ModelNode, ...],
    world_matrices: tuple[Matrix, ...],
    unit_scale: float,
    *,
    positions_already_bound: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if mesh.positions is None:
        raise FormatError("mesh vertex format has no position stream")
    positions = mesh.positions.astype(np.float32, copy=True)
    normals = (
        mesh.normals.astype(np.float32, copy=True)
        if mesh.normals is not None
        else None
    )
    vertex_bones = _vertex_bones(mesh, len(nodes))
    transform_bones = (
        np.full_like(vertex_bones, -1) if positions_already_bound else vertex_bones
    )
    for node_index in np.unique(transform_bones):
        if node_index < 0:
            continue
        selected = transform_bones == node_index
        matrix = np.asarray(world_matrices[int(node_index)], dtype=np.float32)
        positions[selected] = (
            positions[selected] @ matrix[:3, :3].T + matrix[:3, 3]
        )
        if normals is not None:
            normal_matrix = np.linalg.inv(matrix[:3, :3]).T
            normals[selected] = normals[selected] @ normal_matrix.T
    return (
        _to_blender_points(positions, unit_scale),
        _to_blender_vectors(normals) if normals is not None else None,
        vertex_bones,
    )


class TextureLibrary:
    def __init__(self, textures: Iterable[XprTexture] = ()) -> None:
        self._textures: dict[str, XprTexture] = {}
        self._images: dict[tuple[str, int, bool], bpy.types.Image] = {}
        for texture in textures:
            self.add(texture)

    def add(self, texture: XprTexture) -> None:
        self._textures.setdefault(texture.name.casefold(), texture)

    def __iter__(self) -> Iterator[XprTexture]:
        return iter(self._textures.values())

    @property
    def image_count(self) -> int:
        return len(self._images)

    def texture(self, name: str) -> XprTexture | None:
        return self._textures.get(name.casefold())

    def image(
        self,
        name: str,
        *,
        face: int = 0,
        non_color: bool = False,
    ) -> bpy.types.Image | None:
        texture = self.texture(name)
        if texture is None:
            return None
        key = (texture.name.casefold(), face, non_color)
        existing = self._images.get(key)
        if existing is not None:
            return existing

        rgba = decode_texture(texture, face)
        if non_color:
            rgba = rgba.copy()
            rgba[:, :, 1] = 255 - rgba[:, :, 1]
        image_name = texture.name if texture.face_count == 1 else f"{texture.name}_face{face}"
        image = bpy.data.images.new(
            image_name,
            width=texture.fetch.width,
            height=texture.fetch.height,
            alpha=True,
            float_buffer=False,
        )
        image.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
        image.alpha_mode = "STRAIGHT"
        pixels = np.flipud(rgba).astype(np.float32) / np.float32(255.0)
        image.pixels.foreach_set(pixels.ravel())
        image["chromehounds_format"] = texture.fetch.format_name
        image["chromehounds_tiled"] = texture.fetch.tiled
        image["chromehounds_endian"] = texture.fetch.endian
        image["chromehounds_swizzle"] = texture.fetch.swizzle
        image["chromehounds_normal_y_converted"] = non_color
        image.use_fake_user = True
        image.update()
        image.pack()
        self._images[key] = image
        return image

    def import_all(self) -> tuple[bpy.types.Image, ...]:
        images: list[bpy.types.Image] = []
        for texture in self._textures.values():
            for face in range(texture.face_count):
                image = self.image(
                    texture.name,
                    face=face,
                    non_color=texture.fetch.format in (FORMAT_DXN, FORMAT_CTX1),
                )
                if image is not None:
                    images.append(image)
        return tuple(images)


def _float4(
    value: object,
    default: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if isinstance(value, tuple) and len(value) == 4:
        return tuple(float(component) for component in value)
    return default


def build_material(
    source: ModelMaterial,
    textures: TextureLibrary,
    name_prefix: str = "",
) -> bpy.types.Material:
    material = bpy.data.materials.new(f"{name_prefix}{source.name}")
    material.use_nodes = True
    material.diffuse_color = _float4(
        source.param_map().get("g_MatDiffCol"),
        (1.0, 1.0, 1.0, 1.0),
    )
    material["chromehounds_shader"] = source.shader
    for param in source.params:
        material[f"chromehounds_{param.name}"] = param.value

    tree = material.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    output.location = (520.0, 0.0)
    principled = tree.nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (240.0, 0.0)
    tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    params = source.param_map()
    diffuse_color = _float4(
        params.get("g_MatDiffCol"),
        (1.0, 1.0, 1.0, 1.0),
    )
    diffuse_coefficient = float(params.get("g_MatDiffCoef", 1.0))
    tinted_diffuse = tuple(
        max(0.0, min(1.0, component * diffuse_coefficient))
        for component in diffuse_color
    )
    principled.inputs["Base Color"].default_value = tinted_diffuse
    specular_power = max(0.0, float(params.get("g_MatSpecPower", 15.0)))
    principled.inputs["Roughness"].default_value = math.sqrt(2.0 / (specular_power + 2.0))
    specular_color = _float4(
        params.get("g_MatSpecCol"),
        (1.0, 1.0, 1.0, 1.0),
    )
    specular_coefficient = max(0.0, float(params.get("g_MatSpecCoef", 1.0)))
    specular_input = principled.inputs.get("Specular IOR Level")
    if specular_input is not None:
        specular_input.default_value = min(
            1.0,
            specular_coefficient * sum(specular_color[:3]) / 3.0 * 0.5,
        )
    principled.inputs["Alpha"].default_value = diffuse_color[3]
    alpha_reference = int(params.get("g_MatAlphaRef", 0))
    alpha_blend_mode = int(params.get("g_MatAlphaBlend", 0))
    additive_blend = alpha_blend_mode == 3 and alpha_reference == 0
    uses_texture_alpha = (
        alpha_reference > 0 or alpha_blend_mode not in (0, 3)
    )
    uses_uniform_alpha = diffuse_color[3] < 0.999
    diffuse_color_socket = None
    diffuse_alpha_socket = None

    diffuse_name = params.get("g_DiffuseTexture")
    if isinstance(diffuse_name, str):
        image = textures.image(diffuse_name)
        if image is not None:
            image_node = tree.nodes.new("ShaderNodeTexImage")
            image_node.name = "Chromehounds Diffuse"
            image_node.label = diffuse_name
            image_node.image = image
            image_node.location = (-280.0, 80.0)
            diffuse_color_socket = image_node.outputs["Color"]
            diffuse_alpha_socket = image_node.outputs["Alpha"]
            tint = tinted_diffuse[:3]
            if tint != (1.0, 1.0, 1.0):
                multiply = tree.nodes.new("ShaderNodeMixRGB")
                multiply.name = "Chromehounds Diffuse Tint"
                multiply.blend_type = "MULTIPLY"
                multiply.inputs["Fac"].default_value = 1.0
                multiply.inputs["Color2"].default_value = (*tint, 1.0)
                multiply.location = (0.0, 100.0)
                tree.links.new(image_node.outputs["Color"], multiply.inputs["Color1"])
                tree.links.new(multiply.outputs["Color"], principled.inputs["Base Color"])
                diffuse_color_socket = multiply.outputs["Color"]
            else:
                tree.links.new(image_node.outputs["Color"], principled.inputs["Base Color"])
            if uses_texture_alpha and not additive_blend:
                alpha_socket = image_node.outputs["Alpha"]
                if alpha_reference > 0:
                    alpha_test = tree.nodes.new("ShaderNodeMath")
                    alpha_test.name = "Chromehounds Alpha Test"
                    alpha_test.operation = "GREATER_THAN"
                    alpha_test.inputs[1].default_value = max(
                        0.0,
                        min(1.0, alpha_reference / 255.0),
                    )
                    alpha_test.location = (-20.0, -60.0)
                    tree.links.new(alpha_socket, alpha_test.inputs[0])
                    alpha_socket = alpha_test.outputs[0]
                if uses_uniform_alpha:
                    alpha_multiply = tree.nodes.new("ShaderNodeMath")
                    alpha_multiply.operation = "MULTIPLY"
                    alpha_multiply.inputs[1].default_value = diffuse_color[3]
                    alpha_multiply.location = (0.0, -40.0)
                    tree.links.new(alpha_socket, alpha_multiply.inputs[0])
                    alpha_socket = alpha_multiply.outputs[0]
                tree.links.new(alpha_socket, principled.inputs["Alpha"])
        paint_name = f"{diffuse_name}_c"
        if textures.texture(paint_name) is not None:
            paint_image = textures.image(paint_name)
            paint_node = tree.nodes.new("ShaderNodeTexImage")
            paint_node.name = "Chromehounds Paint Layer"
            paint_node.label = f"{paint_name} (unconnected paint/customization layer)"
            paint_node.image = paint_image
            paint_node.location = (-560.0, 300.0)

    normal_name = params.get("g_BumpmapTexture")
    if isinstance(normal_name, str):
        normal_image = textures.image(normal_name, non_color=True)
        if normal_image is not None:
            image_node = tree.nodes.new("ShaderNodeTexImage")
            image_node.name = "Chromehounds Normal"
            image_node.label = normal_name
            image_node.image = normal_image
            image_node.location = (-280.0, -220.0)
            normal_node = tree.nodes.new("ShaderNodeNormalMap")
            normal_node.location = (0.0, -180.0)
            tree.links.new(image_node.outputs["Color"], normal_node.inputs["Color"])
            tree.links.new(normal_node.outputs["Normal"], principled.inputs["Normal"])

    if additive_blend:
        for link in tuple(tree.links):
            if link.to_node == output and link.to_socket == output.inputs["Surface"]:
                tree.links.remove(link)
        transparent = tree.nodes.new("ShaderNodeBsdfTransparent")
        transparent.name = "Chromehounds Additive Transparency"
        transparent.location = (0.0, 260.0)
        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = "Chromehounds Additive Emission"
        emission.location = (0.0, 380.0)
        emission.inputs["Color"].default_value = (*tinted_diffuse[:3], 1.0)
        emission.inputs["Strength"].default_value = diffuse_color[3]
        add_shader = tree.nodes.new("ShaderNodeAddShader")
        add_shader.name = "Chromehounds Additive Blend"
        add_shader.location = (250.0, 300.0)
        if diffuse_color_socket is not None:
            tree.links.new(diffuse_color_socket, emission.inputs["Color"])
        if diffuse_alpha_socket is not None:
            strength_socket = diffuse_alpha_socket
            if uses_uniform_alpha:
                strength_multiply = tree.nodes.new("ShaderNodeMath")
                strength_multiply.name = "Chromehounds Additive Strength"
                strength_multiply.operation = "MULTIPLY"
                strength_multiply.inputs[1].default_value = diffuse_color[3]
                strength_multiply.location = (-20.0, 220.0)
                tree.links.new(strength_socket, strength_multiply.inputs[0])
                strength_socket = strength_multiply.outputs[0]
            tree.links.new(strength_socket, emission.inputs["Strength"])
        tree.links.new(transparent.outputs["BSDF"], add_shader.inputs[0])
        tree.links.new(emission.outputs["Emission"], add_shader.inputs[1])
        tree.links.new(add_shader.outputs["Shader"], output.inputs["Surface"])
        material["chromehounds_alpha_mapping"] = "ADDITIVE"
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "BLENDED"
        elif hasattr(material, "blend_method"):
            material.blend_method = "BLEND"
    elif alpha_reference > 0 or alpha_blend_mode or uses_uniform_alpha:
        material["chromehounds_alpha_mapping"] = (
            "CUTOUT" if alpha_reference > 0 else "BLEND"
        )
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = (
                "DITHERED" if alpha_reference > 0 else "BLENDED"
            )
        elif hasattr(material, "blend_method"):
            material.blend_method = "HASHED" if alpha_reference > 0 else "BLEND"
        if hasattr(material, "alpha_threshold"):
            material.alpha_threshold = max(0.0, min(1.0, alpha_reference / 255.0))
    else:
        material["chromehounds_alpha_mapping"] = "OPAQUE"
        principled.inputs["Alpha"].default_value = 1.0
        material.diffuse_color = (*material.diffuse_color[:3], 1.0)
        if hasattr(material, "use_transparency_overlap"):
            material.use_transparency_overlap = False
        if hasattr(material, "use_transparent_shadow"):
            material.use_transparent_shadow = False
    return material


def _create_mesh_data(
    name: str,
    mesh: ModelMesh,
    positions: np.ndarray,
    normals: np.ndarray | None,
    *,
    flip_winding: bool,
) -> bpy.types.Mesh:
    triangles = mesh.triangles(flip_winding=flip_winding)
    if len(triangles) and int(triangles.max()) >= len(positions):
        raise FormatError(
            f"mesh {name!r} references vertex {int(triangles.max())}, "
            f"but has only {len(positions)} vertices"
        )
    data = bpy.data.meshes.new(name)
    data.vertices.add(len(positions))
    data.vertices.foreach_set("co", positions.ravel())
    loop_vertices = triangles.ravel().astype(np.int32, copy=False)
    data.loops.add(len(loop_vertices))
    data.loops.foreach_set("vertex_index", loop_vertices)
    data.polygons.add(len(triangles))
    data.polygons.foreach_set(
        "loop_start",
        np.arange(0, len(loop_vertices), 3, dtype=np.int32),
    )
    data.polygons.foreach_set(
        "loop_total",
        np.full(len(triangles), 3, dtype=np.int32),
    )
    data.polygons.foreach_set(
        "use_smooth",
        np.ones(len(triangles), dtype=np.bool_),
    )
    data.update(calc_edges=True)

    if normals is not None:
        data.normals_split_custom_set_from_vertices(normals)
    for index, uv_values in enumerate(mesh.uvs):
        if len(uv_values) != len(positions):
            continue
        uv_layer = data.uv_layers.new(name=f"UVMap{index}" if index else "UVMap")
        converted = uv_values.astype(np.float32, copy=True)
        converted[:, 1] = 1.0 - converted[:, 1]
        loop_uvs = converted[loop_vertices]
        if hasattr(uv_layer, "uv"):
            uv_layer.uv.foreach_set("vector", loop_uvs.ravel())
        else:
            uv_layer.data.foreach_set("uv", loop_uvs.ravel())
    if mesh.colors is not None and len(mesh.colors) == len(positions):
        color = data.color_attributes.new(
            name="Color",
            type="FLOAT_COLOR",
            domain="POINT",
        )
        color.data.foreach_set("color", mesh.colors.astype(np.float32).ravel())
    return data


def _add_weights(
    obj: bpy.types.Object,
    mesh: ModelMesh,
    nodes: tuple[ModelNode, ...],
    vertex_bones: np.ndarray,
) -> None:
    groups = [obj.vertex_groups.new(name=node.name) for node in nodes]
    if mesh.bone_indices is not None:
        weights = (
            mesh.bone_weights.astype(np.float32, copy=True)
            if mesh.bone_weights is not None
            else np.ones_like(mesh.bone_indices, dtype=np.float32)
        )
        totals = np.zeros(len(mesh.bone_indices), dtype=np.float32)
        resolved: list[np.ndarray] = []
        for slot in range(mesh.bone_indices.shape[1]):
            node_indices = np.asarray(
                [
                    _palette_bone(mesh, int(index), len(nodes))
                    for index in mesh.bone_indices[:, slot]
                ],
                dtype=np.int32,
            )
            resolved.append(node_indices)
            totals += np.where(node_indices >= 0, np.maximum(weights[:, slot], 0.0), 0.0)
        totals[totals <= 1e-8] = 1.0
        for slot, node_indices in enumerate(resolved):
            for node_index in np.unique(node_indices):
                if node_index < 0:
                    continue
                selected = np.flatnonzero(
                    (node_indices == node_index) & (weights[:, slot] > 1e-8)
                )
                for vertex_index in selected:
                    groups[int(node_index)].add(
                        [int(vertex_index)],
                        float(weights[vertex_index, slot] / totals[vertex_index]),
                        "ADD",
                    )
    else:
        for node_index in np.unique(vertex_bones):
            if node_index >= 0:
                groups[int(node_index)].add(
                    np.flatnonzero(vertex_bones == node_index).tolist(),
                    1.0,
                    "REPLACE",
                )


def _build_armature(
    name: str,
    nodes: tuple[ModelNode, ...],
    world_matrices: tuple[Matrix, ...],
    collection: bpy.types.Collection,
    unit_scale: float,
) -> bpy.types.Object:
    armature_data = bpy.data.armatures.new(f"{name}_Armature")
    armature = bpy.data.objects.new(name, armature_data)
    armature.show_in_front = True
    armature.data.display_type = "STICK"
    collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones: list[bpy.types.EditBone] = []
    for index, node in enumerate(nodes):
        bone = armature_data.edit_bones.new(node.name or f"Bone_{index:03d}")
        edit_bones.append(bone)
    for index, (node, bone) in enumerate(zip(nodes, edit_bones, strict=True)):
        if node.parent_index >= 0:
            bone.parent = edit_bones[node.parent_index]
            bone.use_connect = False
        child_offsets = [
            Vector(nodes[child].translation).length * unit_scale
            for child in range(len(nodes))
            if nodes[child].parent_index == index
            and Vector(nodes[child].translation).length > 1e-4
        ]
        bone.length = max(0.01, min(child_offsets, default=10.0 * unit_scale))
        bone.matrix = _to_blender_matrix(world_matrices[index], unit_scale)
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.select_set(False)
    return armature


def _build_dummies(
    model: Mdl4,
    armature: bpy.types.Object,
    collection: bpy.types.Collection,
    unit_scale: float,
) -> None:
    for index, dummy in enumerate(model.dummies):
        empty = bpy.data.objects.new(
            f"Dummy_{dummy.reference_id:04d}_{index:03d}",
            None,
        )
        empty.empty_display_type = "ARROWS"
        empty.empty_display_size = max(0.01, 10.0 * unit_scale)
        empty.location = Vector(
            (
                dummy.position[0] * unit_scale,
                -dummy.position[2] * unit_scale,
                dummy.position[1] * unit_scale,
            )
        )
        forward = Vector((dummy.forward[0], -dummy.forward[2], dummy.forward[1]))
        if forward.length > 1e-8:
            empty.rotation_mode = "QUATERNION"
            empty.rotation_quaternion = Vector((0.0, 1.0, 0.0)).rotation_difference(
                forward.normalized()
            )
        empty.color = tuple(component / 255.0 for component in dummy.color)
        empty["chromehounds_reference_id"] = dummy.reference_id
        empty["chromehounds_parent_bone_index"] = dummy.parent_bone_index
        empty["chromehounds_attach_bone_index"] = dummy.attach_bone_index
        collection.objects.link(empty)
        if 0 <= dummy.parent_bone_index < len(model.nodes):
            world = empty.matrix_world.copy()
            empty.parent = armature
            empty.parent_type = "BONE"
            empty.parent_bone = model.nodes[dummy.parent_bone_index].name
            empty.matrix_world = world
        else:
            empty.parent = armature


def build_model(
    model: Mdl4 | Smd4,
    name: str,
    parent_collection: bpy.types.Collection,
    textures: TextureLibrary,
    *,
    unit_scale: float = 0.01,
    create_armature: bool = True,
    material_prefix: str = "",
    double_sided: bool = False,
) -> BuiltModel:
    collection = bpy.data.collections.new(name)
    parent_collection.children.link(collection)
    world_matrices = _world_matrices(model.nodes)
    armature = (
        _build_armature(name, model.nodes, world_matrices, collection, unit_scale)
        if create_armature and model.nodes
        else None
    )
    if armature is not None:
        root = armature
    else:
        root = bpy.data.objects.new(name, None)
        collection.objects.link(root)

    materials = (
        tuple(
            build_material(material, textures, material_prefix)
            for material in model.materials
        )
        if isinstance(model, Mdl4)
        else ()
    )
    mesh_objects: list[bpy.types.Object] = []
    for mesh_index, source_mesh in enumerate(model.meshes):
        if source_mesh.positions is None:
            continue
        positions, normals, vertex_bones = _bind_mesh(
            source_mesh,
            model.nodes,
            world_matrices,
            unit_scale,
            positions_already_bound=(
                isinstance(model, Smd4) and source_mesh.vertex_format == 2
            ),
        )
        mesh_name = f"{name}_Mesh_{mesh_index:03d}"
        mesh_data = _create_mesh_data(
            mesh_name,
            source_mesh,
            positions,
            normals,
            flip_winding=isinstance(model, Mdl4),
        )
        mesh_object = bpy.data.objects.new(mesh_name, mesh_data)
        collection.objects.link(mesh_object)
        mesh_object.parent = root
        mesh_object["chromehounds_vertex_format"] = source_mesh.vertex_format
        mesh_object["chromehounds_default_bone"] = source_mesh.default_bone_index
        if isinstance(model, Mdl4) and 0 <= source_mesh.material_index < len(materials):
            material = materials[source_mesh.material_index]
            if hasattr(material, "use_backface_culling"):
                material.use_backface_culling = (
                    False if double_sided else source_mesh.cull_backfaces
                )
            mesh_data.materials.append(material)
            mesh_object.show_transparent = (
                material.get("chromehounds_alpha_mapping") != "OPAQUE"
            )
        if armature is not None:
            _add_weights(mesh_object, source_mesh, model.nodes, vertex_bones)
            modifier = mesh_object.modifiers.new("Chromehounds Armature", "ARMATURE")
            modifier.object = armature
        mesh_objects.append(mesh_object)

    if isinstance(model, Mdl4) and armature is not None:
        _build_dummies(model, armature, collection, unit_scale)
    return BuiltModel(
        collection=collection,
        root=root,
        armature=armature,
        mesh_objects=tuple(mesh_objects),
        actions=(),
    )


def build_actions(
    animation: M2aAnimation,
    model: Mdl4,
    built: BuiltModel,
    *,
    unit_scale: float = 0.01,
    name_prefix: str = "",
) -> tuple[bpy.types.Action, ...]:
    armature = built.armature
    if armature is None:
        raise FormatError("cannot import M2A animation without an armature")
    if animation.node_count != len(model.nodes):
        raise FormatError(
            f"M2A contains {animation.node_count} tracks per clip, "
            f"but model has {len(model.nodes)} nodes"
        )
    rest_local = tuple(
        _to_blender_matrix(
            _source_local_matrix(node.translation, node.rotation, node.scale),
            unit_scale,
        )
        for node in model.nodes
    )
    animation_data = armature.animation_data_create()
    actions: list[bpy.types.Action] = []
    for clip in animation.clips:
        action = bpy.data.actions.new(f"{name_prefix}Clip_{clip.clip_id:03d}")
        action.use_fake_user = True
        action["chromehounds_clip_id"] = clip.clip_id
        action["chromehounds_duration"] = clip.duration
        animation_data.action = action
        for node_index, track in enumerate(clip.tracks):
            if not track.keys:
                continue
            pose_bone = armature.pose.bones.get(model.nodes[node_index].name)
            if pose_bone is None:
                continue
            pose_bone.rotation_mode = "QUATERNION"
            for key in track.keys:
                animated_local = _to_blender_matrix(
                    _source_local_matrix(key.translation, key.rotation, key.scale),
                    unit_scale,
                )
                basis = rest_local[node_index].inverted_safe() @ animated_local
                location, rotation, scale = basis.decompose()
                pose_bone.location = location
                pose_bone.rotation_quaternion = Quaternion(rotation)
                pose_bone.scale = scale
                pose_bone.keyframe_insert(
                    data_path="location",
                    frame=key.frame,
                    group=pose_bone.name,
                )
                pose_bone.keyframe_insert(
                    data_path="rotation_quaternion",
                    frame=key.frame,
                    group=pose_bone.name,
                )
                pose_bone.keyframe_insert(
                    data_path="scale",
                    frame=key.frame,
                    group=pose_bone.name,
                )
        actions.append(action)
    animation_data.action = None
    for pose_bone in armature.pose.bones:
        pose_bone.matrix_basis.identity()
    built.actions = tuple(actions)
    return built.actions


def map_transform(
    position: tuple[float, float, float],
    rotation: tuple[float, float, float],
    scale: tuple[float, float, float],
    unit_scale: float,
) -> Matrix:
    source = _source_local_matrix(
        position,
        tuple(math.radians(value) for value in rotation),
        scale,
    )
    return _to_blender_matrix(source, unit_scale)
