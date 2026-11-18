from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .binary import BinaryReader
from .errors import FormatError


@dataclass(frozen=True, slots=True)
class Dummy:
    position: tuple[float, float, float]
    forward: tuple[float, float, float]
    color: tuple[int, int, int, int]
    reference_id: int
    parent_bone_index: int
    attach_bone_index: int


@dataclass(frozen=True, slots=True)
class MaterialParam:
    kind: int
    name: str
    value: int | float | tuple[float, float, float, float] | str


@dataclass(frozen=True, slots=True)
class ModelMaterial:
    name: str
    shader: str
    flags: tuple[int, int, int]
    params: tuple[MaterialParam, ...]

    def param_map(self) -> dict[str, int | float | tuple[float, float, float, float] | str]:
        return {param.name: param.value for param in self.params}


@dataclass(frozen=True, slots=True)
class ModelNode:
    name: str
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float]
    scale: tuple[float, float, float]
    bounding_box_min: tuple[float, float, float]
    bounding_box_max: tuple[float, float, float]
    parent_index: int
    child_index: int
    next_sibling_index: int
    previous_sibling_index: int


@dataclass(slots=True)
class ModelMesh:
    vertex_format: int
    material_index: int
    cull_backfaces: bool
    triangle_strip: bool
    default_bone_index: int
    bone_palette: np.ndarray
    indices: np.ndarray
    positions: np.ndarray | None
    normals: np.ndarray | None
    normal_bone_indices: np.ndarray | None
    tangents: np.ndarray | None
    bitangents: np.ndarray | None
    colors: np.ndarray | None
    uvs: tuple[np.ndarray, ...]
    bone_indices: np.ndarray | None
    bone_weights: np.ndarray | None
    unknown_blocks: tuple[bytes, ...] = field(default_factory=tuple)

    def triangles(self, flip_winding: bool = False) -> np.ndarray:
        return triangulate_strip(self.indices, flip_winding=flip_winding)


@dataclass(frozen=True, slots=True)
class Mdl4:
    version: int
    bounding_box_min: tuple[float, float, float]
    bounding_box_max: tuple[float, float, float]
    dummies: tuple[Dummy, ...]
    materials: tuple[ModelMaterial, ...]
    nodes: tuple[ModelNode, ...]
    meshes: tuple[ModelMesh, ...]


@dataclass(frozen=True, slots=True)
class Smd4:
    version: int
    bounding_box_min: tuple[float, float, float]
    bounding_box_max: tuple[float, float, float]
    nodes: tuple[ModelNode, ...]
    meshes: tuple[ModelMesh, ...]


def _vec3(reader: BinaryReader) -> tuple[float, float, float]:
    return reader.unpack("3f", "vector3")


def _sign_extend(values: np.ndarray, bits: int) -> np.ndarray:
    sign = 1 << (bits - 1)
    return ((values ^ sign) - sign).astype(np.int32, copy=False)


def _decode_packed_vector(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = values.astype(np.uint32, copy=False)
    x = _sign_extend(raw & 0x3FF, 10)
    y = _sign_extend((raw >> 10) & 0x3FF, 10)
    z = _sign_extend((raw >> 20) & 0x3FF, 10)
    w = _sign_extend((raw >> 30) & 0x3, 2)
    xyz = np.stack((x, y, z), axis=1).astype(np.float32) / np.float32(511.0)
    lengths = np.linalg.norm(xyz, axis=1)
    nonzero = lengths > 1e-8
    xyz[nonzero] /= lengths[nonzero, None]
    return xyz, w


def _read_param(reader: BinaryReader, offset: int) -> MaterialParam:
    reader.require(offset, 0x40, "material parameter")
    kind = reader.data[offset]
    name_reader = BinaryReader(reader.data, ">", reader.label, offset + 1)
    name = name_reader.fixed_string(0x1F, what="material parameter name")
    value_offset = offset + 0x20
    if kind == 0:
        value = reader.unpack_from(">i", value_offset, "material int parameter")[0]
    elif kind == 1:
        value = reader.unpack_from(">f", value_offset, "material float parameter")[0]
    elif kind == 4:
        value = reader.unpack_from(">4f", value_offset, "material float4 parameter")
    elif kind == 5:
        value = reader.cstring_at(
            value_offset,
            max_size=0x20,
            what="material string parameter",
        )
    else:
        raise FormatError(
            f"{reader.label}: unsupported material parameter type {kind} at 0x{offset:X}"
        )
    return MaterialParam(kind=kind, name=name, value=value)


def _read_material(reader: BinaryReader, offset: int) -> ModelMaterial:
    reader.require(offset, 0x840, "MDL4 material")
    local = BinaryReader(reader.data, ">", reader.label, offset)
    name = local.fixed_string(0x1F, what="material name")
    shader = local.fixed_string(0x1D, what="shader name")
    flags = (local.u8(), local.u8(), local.u8())
    param_count = local.u8()
    if param_count > 32:
        raise FormatError(f"{reader.label}: unreasonable material parameter count {param_count}")
    params = tuple(_read_param(reader, offset + 0x40 + index * 0x40) for index in range(param_count))
    return ModelMaterial(name=name, shader=shader, flags=flags, params=params)


def _read_node(reader: BinaryReader, offset: int, smd: bool = False) -> ModelNode:
    reader.require(offset, 0x90, "model node")
    local = BinaryReader(reader.data, ">", reader.label, offset)
    name = local.fixed_string(0x20, what="node name")
    translation = _vec3(local)
    rotation = _vec3(local)
    scale = _vec3(local)
    bounds_min = _vec3(local)
    bounds_max = _vec3(local)
    parent, child, next_sibling, previous_sibling = local.unpack("4h", "node hierarchy")
    local.skip(12)
    local.skip(32)
    return ModelNode(
        name=name,
        translation=translation,
        rotation=rotation,
        scale=scale,
        bounding_box_min=bounds_min,
        bounding_box_max=bounds_max,
        parent_index=parent,
        child_index=child,
        next_sibling_index=next_sibling,
        previous_sibling_index=previous_sibling,
    )


def _mesh_vertex_size(version: int, vertex_format: int, smd: bool = False) -> int:
    if smd and version == 0x40001:
        sizes = {0: 0x10, 1: 0x18, 2: 0x24}
    elif version == 0x40001:
        sizes = {0: 0x40, 1: 0x54, 2: 0x3C}
    elif version == 0x40002:
        sizes = {0: 0x28, 1: 0x34}
    else:
        raise FormatError(f"unsupported model version 0x{version:X}")
    try:
        return sizes[vertex_format]
    except KeyError as exc:
        raise FormatError(
            f"unsupported vertex format {vertex_format} for model version 0x{version:X}"
        ) from exc


def _empty_vertex_fields() -> dict:
    return {
        "positions": None,
        "normals": None,
        "normal_bone_indices": None,
        "tangents": None,
        "bitangents": None,
        "colors": None,
        "uvs": (),
        "bone_indices": None,
        "bone_weights": None,
    }


def _read_mdl_vertices(
    reader: BinaryReader,
    offset: int,
    count: int,
    version: int,
    vertex_format: int,
) -> dict:
    size = _mesh_vertex_size(version, vertex_format)
    reader.require(offset, count * size, "MDL4 vertex buffer")
    raw = reader.data[offset : offset + count * size]
    fields = _empty_vertex_fields()

    if version == 0x40001 and vertex_format in (0, 1):
        dtype_fields: list[tuple] = [
            ("position", ">f4", (3,)),
            ("normal", ">u4"),
            ("tangent", ">u4"),
            ("bitangent", ">u4"),
            ("color", "u1", (4,)),
            ("uv", ">f4", (4, 2)),
        ]
        if vertex_format == 0:
            dtype_fields.extend((("unknown", ">i2"), ("padding", ">i2")))
        else:
            dtype_fields.extend((("bone_indices", ">i2", (4,)), ("bone_weights", ">f4", (4,))))
        array = np.frombuffer(raw, dtype=np.dtype(dtype_fields), count=count)
        fields["positions"] = array["position"].astype(np.float32, copy=True)
        fields["normals"], fields["normal_bone_indices"] = _decode_packed_vector(array["normal"])
        fields["tangents"], _ = _decode_packed_vector(array["tangent"])
        fields["bitangents"], _ = _decode_packed_vector(array["bitangent"])
        argb = array["color"].astype(np.float32) / np.float32(255.0)
        fields["colors"] = argb[:, [1, 2, 3, 0]]
        uv_values = array["uv"].astype(np.float32, copy=True)
        fields["uvs"] = tuple(uv_values[:, index, :] for index in range(4))
        if vertex_format == 1:
            fields["bone_indices"] = array["bone_indices"].astype(np.int16, copy=True)
            fields["bone_weights"] = array["bone_weights"].astype(np.float32, copy=True)
    elif version == 0x40001 and vertex_format == 2:
        dtype = np.dtype(
            [
                ("color", "u1", (4,)),
                ("uv", ">f4", (4, 2)),
                ("bone_indices", ">i2", (4,)),
                ("bone_weights", ">f4", (4,)),
            ]
        )
        array = np.frombuffer(raw, dtype=dtype, count=count)
        argb = array["color"].astype(np.float32) / np.float32(255.0)
        fields["colors"] = argb[:, [1, 2, 3, 0]]
        uv_values = array["uv"].astype(np.float32, copy=True)
        fields["uvs"] = tuple(uv_values[:, index, :] for index in range(4))
        fields["bone_indices"] = array["bone_indices"].astype(np.int16, copy=True)
        fields["bone_weights"] = array["bone_weights"].astype(np.float32, copy=True)
    elif version == 0x40002 and vertex_format in (0, 1):
        dtype_fields = [
            ("position", ">f4", (3,)),
            ("normal", "i1", (4,)),
            ("tangent", "i1", (4,)),
            ("color", "u1", (4,)),
            ("uv", ">i2", (4, 2)),
        ]
        if vertex_format == 1:
            dtype_fields.extend((("bone_indices", "u1", (4,)), ("bone_weights", ">u2", (4,))))
        array = np.frombuffer(raw, dtype=np.dtype(dtype_fields), count=count)
        fields["positions"] = array["position"].astype(np.float32, copy=True)
        packed_normals = array["normal"].astype(np.float32)
        fields["normal_bone_indices"] = array["normal"][:, 0].astype(np.int16)
        fields["normals"] = packed_normals[:, [3, 2, 1]] / np.float32(127.0)
        packed_tangents = array["tangent"].astype(np.float32)
        fields["tangents"] = packed_tangents[:, [3, 2, 1]] / np.float32(127.0)
        argb = array["color"].astype(np.float32) / np.float32(255.0)
        fields["colors"] = argb[:, [1, 2, 3, 0]]
        uv_values = array["uv"].astype(np.float32) / np.float32(2048.0)
        fields["uvs"] = tuple(uv_values[:, index, :] for index in range(4))
        if vertex_format == 1:
            fields["bone_indices"] = array["bone_indices"].astype(np.int16, copy=True)
            fields["bone_weights"] = (
                array["bone_weights"].astype(np.float32) / np.float32(65535.0)
            )
    return fields


def _read_smd_vertices(
    reader: BinaryReader,
    offset: int,
    count: int,
    version: int,
    vertex_format: int,
) -> dict:
    size = _mesh_vertex_size(version, vertex_format, smd=True)
    reader.require(offset, count * size, "SMD4 vertex buffer")
    raw = reader.data[offset : offset + count * size]
    fields = _empty_vertex_fields()
    if vertex_format == 0:
        dtype = np.dtype([("position", ">f4", (3,)), ("bone", ">i2"), ("padding", ">i2")])
        array = np.frombuffer(raw, dtype=dtype, count=count)
        fields["positions"] = array["position"].astype(np.float32, copy=True)
        fields["bone_indices"] = np.full((count, 4), -1, dtype=np.int16)
        fields["bone_indices"][:, 0] = array["bone"].astype(np.int16)
        fields["bone_weights"] = np.zeros((count, 4), dtype=np.float32)
        fields["bone_weights"][:, 0] = 1.0
    elif vertex_format == 1:
        dtype = np.dtype(
            [
                ("position", ">f4", (3,)),
                ("uv", ">f4", (2,)),
                ("bone", ">i2"),
                ("padding", ">i2"),
            ]
        )
        array = np.frombuffer(raw, dtype=dtype, count=count)
        fields["positions"] = array["position"].astype(np.float32, copy=True)
        fields["uvs"] = (array["uv"].astype(np.float32, copy=True),)
        fields["bone_indices"] = np.full((count, 4), -1, dtype=np.int16)
        fields["bone_indices"][:, 0] = array["bone"].astype(np.int16)
        fields["bone_weights"] = np.zeros((count, 4), dtype=np.float32)
        fields["bone_weights"][:, 0] = 1.0
    else:
        dtype = np.dtype(
            [
                ("position", ">f4", (3,)),
                ("bone_indices", ">i2", (4,)),
                ("bone_weights", ">f4", (4,)),
            ]
        )
        array = np.frombuffer(raw, dtype=dtype, count=count)
        fields["positions"] = array["position"].astype(np.float32, copy=True)
        fields["bone_indices"] = array["bone_indices"].astype(np.int16, copy=True)
        fields["bone_weights"] = array["bone_weights"].astype(np.float32, copy=True)
    return fields


def _read_mesh(
    reader: BinaryReader,
    offset: int,
    data_offset: int,
    version: int,
    smd: bool = False,
) -> tuple[ModelMesh, int]:
    local = BinaryReader(reader.data, ">", reader.label, offset)
    vertex_format = local.u8()
    material_index = local.u8()
    cull_backfaces = bool(local.u8())
    triangle_strip = bool(local.u8())
    index_count = local.u16()
    default_bone_index = local.i16()
    bone_palette = np.frombuffer(
        local.bytes(56, "mesh bone palette"),
        dtype=">i2",
        count=28,
    ).astype(np.int16)
    index_buffer_size = local.u32()
    index_buffer_offset = local.u32()
    vertex_buffer_size = local.u32()
    vertex_buffer_offset = local.u32()

    unknown_blocks: list[bytes] = []
    if not smd and vertex_format == 2:
        for block_index in range(16):
            length = local.u32()
            block_offset = local.u32()
            unknown_blocks.append(
                reader.bytes_at(
                    data_offset + block_offset,
                    length,
                    f"mesh unknown block {block_index}",
                )
            )

    if index_buffer_size < index_count * 2:
        raise FormatError(
            f"{reader.label}: mesh index buffer is {index_buffer_size} bytes for "
            f"{index_count} 16-bit indices"
        )
    raw_indices = reader.bytes_at(
        data_offset + index_buffer_offset,
        index_count * 2,
        "mesh index buffer",
    )
    indices = np.frombuffer(raw_indices, dtype=">u2").astype(np.uint16)
    vertex_size = _mesh_vertex_size(version, vertex_format, smd)
    if vertex_buffer_size % vertex_size:
        raise FormatError(
            f"{reader.label}: mesh vertex buffer size 0x{vertex_buffer_size:X} "
            f"is not divisible by stride 0x{vertex_size:X}"
        )
    vertex_count = vertex_buffer_size // vertex_size
    fields = (
        _read_smd_vertices(
            reader,
            data_offset + vertex_buffer_offset,
            vertex_count,
            version,
            vertex_format,
        )
        if smd
        else _read_mdl_vertices(
            reader,
            data_offset + vertex_buffer_offset,
            vertex_count,
            version,
            vertex_format,
        )
    )
    mesh = ModelMesh(
        vertex_format=vertex_format,
        material_index=material_index,
        cull_backfaces=cull_backfaces,
        triangle_strip=triangle_strip,
        default_bone_index=default_bone_index,
        bone_palette=bone_palette,
        indices=indices,
        unknown_blocks=tuple(unknown_blocks),
        **fields,
    )
    return mesh, local.position - offset


def parse_mdl4(data: bytes, label: str = "<mdl4>") -> Mdl4:
    reader = BinaryReader(data, ">", label)
    if len(reader) < 0x80 or reader.bytes_at(0, 4) != b"MDL4":
        raise FormatError(f"{label}: expected an MDL4 model")
    reader.seek(4)
    version = reader.u32()
    if version not in (0x40001, 0x40002):
        raise FormatError(f"{label}: unsupported MDL4 version 0x{version:X}")
    data_offset = reader.u32()
    data_length = reader.u32()
    dummy_count, material_count, node_count, mesh_count, _buffer_count = reader.unpack(
        "5I", "MDL4 counts"
    )
    bounds_min = _vec3(reader)
    bounds_max = _vec3(reader)
    reader.skip(8)
    reader.skip(0x3C)
    reader.require(data_offset, data_length, "MDL4 data section")

    if any(count > 100_000 for count in (dummy_count, material_count, node_count, mesh_count)):
        raise FormatError(f"{label}: unreasonable MDL4 object counts")

    dummies: list[Dummy] = []
    for _ in range(dummy_count):
        position = _vec3(reader)
        forward = _vec3(reader)
        color = reader.unpack("4B", "dummy color")
        reference_id, parent_index, attach_index, _unknown = reader.unpack(
            "4h", "dummy metadata"
        )
        reader.skip(12)
        dummies.append(
            Dummy(
                position=position,
                forward=forward,
                color=color,
                reference_id=reference_id,
                parent_bone_index=parent_index,
                attach_bone_index=attach_index,
            )
        )

    materials = tuple(
        _read_material(reader, reader.position + index * 0x840)
        for index in range(material_count)
    )
    reader.skip(material_count * 0x840)
    nodes = tuple(_read_node(reader, reader.position + index * 0x90) for index in range(node_count))
    reader.skip(node_count * 0x90)
    meshes: list[ModelMesh] = []
    for _ in range(mesh_count):
        mesh, consumed = _read_mesh(reader, reader.position, data_offset, version)
        meshes.append(mesh)
        reader.skip(consumed)
    return Mdl4(
        version=version,
        bounding_box_min=bounds_min,
        bounding_box_max=bounds_max,
        dummies=tuple(dummies),
        materials=materials,
        nodes=nodes,
        meshes=tuple(meshes),
    )


def parse_smd4(data: bytes, label: str = "<smd4>") -> Smd4:
    reader = BinaryReader(data, ">", label)
    if len(reader) < 0x60 or reader.bytes_at(0, 4) != b"SMD4":
        raise FormatError(f"{label}: expected an SMD4 model")
    reader.seek(4)
    version = reader.u32()
    if version != 0x40001:
        raise FormatError(f"{label}: unsupported SMD4 version 0x{version:X}")
    data_offset = reader.u32()
    data_size = reader.u32()
    unknown_count, node_count, mesh_count, _buffer_count = reader.unpack("4I", "SMD4 counts")
    bounds_min = _vec3(reader)
    bounds_max = _vec3(reader)
    reader.skip(8)
    reader.skip(0x20)
    reader.require(data_offset, data_size, "SMD4 data section")
    reader.skip(unknown_count * 0x24)
    nodes = tuple(
        _read_node(reader, reader.position + index * 0x90, smd=True)
        for index in range(node_count)
    )
    reader.skip(node_count * 0x90)
    meshes: list[ModelMesh] = []
    for _ in range(mesh_count):
        mesh, consumed = _read_mesh(reader, reader.position, data_offset, version, smd=True)
        meshes.append(mesh)
        reader.skip(consumed)
    return Smd4(
        version=version,
        bounding_box_min=bounds_min,
        bounding_box_max=bounds_max,
        nodes=nodes,
        meshes=tuple(meshes),
    )


def triangulate_strip(
    indices: np.ndarray,
    *,
    flip_winding: bool = False,
) -> np.ndarray:
    source = np.asarray(indices, dtype=np.uint16)
    triangles: list[tuple[int, int, int]] = []
    flip = flip_winding
    for index in range(max(0, len(source) - 2)):
        a, b, c = (int(value) for value in source[index : index + 3])
        if 0xFFFF in (a, b, c):
            flip = flip_winding
            continue
        if a != b and b != c and c != a:
            triangles.append((c, b, a) if flip else (a, b, c))
        flip = not flip
    if not triangles:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(triangles, dtype=np.int32)
