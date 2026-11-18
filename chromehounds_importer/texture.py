from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binary import BinaryReader
from .errors import FormatError

FORMAT_8 = 2
FORMAT_1555 = 3
FORMAT_565 = 4
FORMAT_8888 = 6
FORMAT_88 = 10
FORMAT_4444 = 15
FORMAT_DXT1 = 18
FORMAT_DXT3 = 19
FORMAT_DXT5 = 20
FORMAT_RGBA16F = 32
FORMAT_DXN = 49
FORMAT_DXT3A = 58
FORMAT_DXT5A = 59
FORMAT_CTX1 = 60
FORMAT_DXT3A_1111 = 61

_AS_16_BASE = (6, 18, 19, 20, 7, 16, 17)
_FORMAT_INFO = {
    FORMAT_8: (1, 1),
    FORMAT_1555: (1, 2),
    FORMAT_565: (1, 2),
    FORMAT_8888: (1, 4),
    FORMAT_88: (1, 2),
    FORMAT_4444: (1, 2),
    FORMAT_DXT1: (4, 8),
    FORMAT_DXT3: (4, 16),
    FORMAT_DXT5: (4, 16),
    FORMAT_RGBA16F: (1, 8),
    FORMAT_DXN: (4, 16),
    FORMAT_DXT3A: (4, 8),
    FORMAT_DXT5A: (4, 8),
    FORMAT_CTX1: (4, 8),
    FORMAT_DXT3A_1111: (4, 8),
}
_FORMAT_NAMES = {
    FORMAT_8: "R8",
    FORMAT_1555: "ARGB1555",
    FORMAT_565: "RGB565",
    FORMAT_8888: "RGBA8",
    FORMAT_88: "RG8",
    FORMAT_4444: "ARGB4444",
    FORMAT_DXT1: "BC1",
    FORMAT_DXT3: "BC2",
    FORMAT_DXT5: "BC3",
    FORMAT_RGBA16F: "RGBA16F",
    FORMAT_DXN: "BC5",
    FORMAT_DXT3A: "DXT3A",
    FORMAT_DXT5A: "BC4",
    FORMAT_CTX1: "CTX1",
    FORMAT_DXT3A_1111: "DXT3A1",
}


@dataclass(frozen=True, slots=True)
class TextureFetch:
    dwords: tuple[int, int, int, int, int, int]

    @property
    def resource_type(self) -> int:
        return self.dwords[0] & 3

    @property
    def tiled(self) -> bool:
        return bool(self.dwords[0] >> 31)

    @property
    def pitch(self) -> int:
        return ((self.dwords[0] >> 22) & 0x1FF) << 5

    @property
    def raw_format(self) -> int:
        return self.dwords[1] & 0x3F

    @property
    def format(self) -> int:
        raw = self.raw_format
        return _AS_16_BASE[raw - 50] if 50 <= raw <= 56 else raw

    @property
    def format_name(self) -> str:
        return _FORMAT_NAMES.get(self.format, f"Xenos_{self.format}")

    @property
    def endian(self) -> int:
        return (self.dwords[1] >> 6) & 3

    @property
    def address(self) -> int:
        return self.dwords[1] & 0xFFFFF000

    @property
    def dimension(self) -> int:
        return (self.dwords[5] >> 9) & 3

    @property
    def width(self) -> int:
        mask = 0x7FF if self.dimension == 2 else 0x1FFF
        return (self.dwords[2] & mask) + 1

    @property
    def height(self) -> int:
        if self.dimension == 2:
            return ((self.dwords[2] >> 11) & 0x7FF) + 1
        return ((self.dwords[2] >> 13) & 0x1FFF) + 1

    @property
    def depth(self) -> int:
        return ((self.dwords[2] >> 22) & 0x3FF) + 1 if self.dimension == 2 else 1

    @property
    def swizzle(self) -> int:
        return (self.dwords[3] >> 1) & 0xFFF

    @property
    def packed_mips(self) -> bool:
        return bool((self.dwords[5] >> 11) & 1)

    @property
    def mip_address(self) -> int:
        return self.dwords[5] & 0xFFFFF000


@dataclass(frozen=True, slots=True)
class XprTexture:
    name: str
    tag_type: str
    fetch: TextureFetch
    data: memoryview

    @property
    def face_count(self) -> int:
        return 6 if self.tag_type == "TXCM" else 1


@dataclass(frozen=True, slots=True)
class XprArchive:
    textures: tuple[XprTexture, ...]


def parse_xpr(data: bytes, label: str = "<xpr>") -> XprArchive:
    reader = BinaryReader(data, ">", label)
    if len(reader) < 16 or reader.bytes_at(0, 4) != b"XPR2":
        raise FormatError(f"{label}: expected an XPR2 resource package")
    header_span, data_size = reader.unpack_from(">II", 4, "XPR2 header")
    data_offset = 12 + header_span
    reader.require(data_offset, data_size, "XPR2 data section")
    tag_base = 12
    tag_count = reader.unpack_from(">I", tag_base, "XPR2 tag count")[0]
    if tag_count > 100_000:
        raise FormatError(f"{label}: unreasonable XPR2 tag count {tag_count}")
    reader.require(16, tag_count * 16, "XPR2 tag table")
    payload = reader.data[data_offset : data_offset + data_size]
    textures: list[XprTexture] = []
    for index in range(tag_count):
        offset = 16 + index * 16
        raw_type, resource_offset, resource_size, name_offset = reader.unpack_from(
            ">4sIII", offset, f"XPR2 tag {index}"
        )
        try:
            tag_type = raw_type.decode("ascii")
        except UnicodeDecodeError as exc:
            raise FormatError(f"{label}: invalid XPR2 tag type at index {index}") from exc
        name = reader.cstring_at(
            tag_base + name_offset,
            encoding="cp932",
            what=f"XPR2 tag {index} name",
        )
        if tag_type == "USER":
            continue
        if tag_type not in {"TX2D", "TX3D", "TXCM"}:
            raise FormatError(f"{label}: unsupported XPR2 resource type {tag_type!r}")
        if resource_size < 0x28 or resource_size % 4:
            raise FormatError(
                f"{label}: XPR2 texture {name!r} descriptor is only {resource_size} bytes"
            )
        descriptor_offset = tag_base + resource_offset
        reader.require(descriptor_offset, resource_size, f"XPR2 texture {name} descriptor")
        descriptor = reader.unpack_from(
            f">{resource_size // 4}I",
            descriptor_offset,
            f"XPR2 texture {name} descriptor",
        )
        fetch = TextureFetch(tuple(descriptor[-6:]))
        if fetch.resource_type != 2:
            raise FormatError(
                f"{label}: XPR2 texture {name!r} has resource type {fetch.resource_type}, expected 2"
            )
        expected_dimension = {"TX2D": 1, "TX3D": 2, "TXCM": 3}[tag_type]
        if fetch.dimension != expected_dimension:
            raise FormatError(
                f"{label}: {tag_type} texture {name!r} has Xenos dimension "
                f"{fetch.dimension}, expected {expected_dimension}"
            )
        if fetch.format not in _FORMAT_INFO:
            raise FormatError(
                f"{label}: XPR2 texture {name!r} uses unsupported Xenos format "
                f"{fetch.format} (raw {fetch.raw_format})"
            )
        if not (1 <= fetch.width <= 8192 and 1 <= fetch.height <= 8192):
            raise FormatError(
                f"{label}: XPR2 texture {name!r} has invalid dimensions "
                f"{fetch.width}x{fetch.height}"
            )
        textures.append(XprTexture(name=name, tag_type=tag_type, fetch=fetch, data=payload))
    return XprArchive(textures=tuple(textures))


def _tiled_outer(y: np.ndarray, width: int, log2_bpp: int) -> np.ndarray:
    y = y.astype(np.int64, copy=False)
    macro = ((y >> 5) * (width >> 5)) << (log2_bpp + 7)
    micro = ((y & 6) << 2) << log2_bpp
    return macro + ((micro & ~0xF) << 1) + (micro & 0xF) + ((y & 8) << (3 + log2_bpp)) + ((y & 1) << 4)


def _tiled_inner(
    x: np.ndarray,
    y: np.ndarray,
    log2_bpp: int,
    base_offset: np.ndarray,
) -> np.ndarray:
    x = x.astype(np.int64, copy=False)
    y = y.astype(np.int64, copy=False)
    macro = (x >> 5) << (log2_bpp + 7)
    micro = (x & 7) << log2_bpp
    offset = base_offset + macro + ((micro & ~0xF) << 1) + (micro & 0xF)
    return (
        ((offset & ~0x1FF) << 3)
        + ((offset & 0x1C0) << 2)
        + (offset & 0x3F)
        + ((y & 16) << 7)
        + (((((y & 8) >> 2) + (x >> 3)) & 3) << 6)
    )


def _packed_mip_offset(fetch: TextureFetch, block_size: int) -> tuple[int, int]:
    if not fetch.packed_mips:
        return 0, 0
    width_log = max(0, (fetch.width - 1).bit_length())
    height_log = max(0, (fetch.height - 1).bit_length())
    if min(width_log, height_log) > 4:
        return 0, 0
    if width_log > height_log:
        return 0, 16 // block_size
    return 16 // block_size, 0


def _source_size(fetch: TextureFetch) -> int:
    block_size, block_bytes = _FORMAT_INFO[fetch.format]
    blocks_x = (fetch.width + block_size - 1) // block_size
    blocks_y = (fetch.height + block_size - 1) // block_size
    pitch_blocks = max(fetch.pitch // block_size, blocks_x)
    rows = blocks_y
    if fetch.tiled:
        pitch_blocks = (pitch_blocks + 31) & ~31
        rows = (rows + 31) & ~31
    else:
        _, packed_y = _packed_mip_offset(fetch, block_size)
        rows += packed_y
    return pitch_blocks * rows * block_bytes


def _face_stride(fetch: TextureFetch) -> int:
    return (_source_size(fetch) + 4095) & ~4095


def _swap_endian(blocks: np.ndarray, endian: int) -> np.ndarray:
    result = blocks.copy()
    if endian == 1:
        result = result.reshape(*result.shape[:-1], -1, 2)[..., ::-1].reshape(result.shape)
    elif endian == 2:
        result = result.reshape(*result.shape[:-1], -1, 4)[..., ::-1].reshape(result.shape)
    elif endian == 3:
        words = result.reshape(*result.shape[:-1], -1, 4)
        result = words[..., [2, 3, 0, 1]].reshape(result.shape)
    return result


def _linear_blocks(texture: XprTexture, face: int) -> tuple[np.ndarray, int, int]:
    fetch = texture.fetch
    block_size, block_bytes = _FORMAT_INFO[fetch.format]
    blocks_x = (fetch.width + block_size - 1) // block_size
    blocks_y = (fetch.height + block_size - 1) // block_size
    pitch_blocks = max(fetch.pitch // block_size, blocks_x)
    if fetch.tiled:
        pitch_blocks = (pitch_blocks + 31) & ~31
    pack_x, pack_y = _packed_mip_offset(fetch, block_size)
    x, y = np.meshgrid(
        np.arange(blocks_x, dtype=np.int64) + pack_x,
        np.arange(blocks_y, dtype=np.int64) + pack_y,
    )
    if fetch.tiled:
        outer = _tiled_outer(y, pitch_blocks, block_bytes.bit_length() - 1)
        offsets = _tiled_inner(
            x,
            y,
            block_bytes.bit_length() - 1,
            outer,
        )
    else:
        offsets = (y * pitch_blocks + x) * block_bytes
    base = fetch.address + face * _face_stride(fetch)
    offsets += base
    byte_indices = offsets[..., None] + np.arange(block_bytes, dtype=np.int64)
    maximum = int(byte_indices.max(initial=0))
    if maximum >= len(texture.data):
        raise FormatError(
            f"XPR2 texture {texture.name!r} face {face} reads byte 0x{maximum:X}, "
            f"outside its 0x{len(texture.data):X}-byte data section"
        )
    source = np.frombuffer(texture.data, dtype=np.uint8)
    blocks = source[byte_indices]
    return _swap_endian(blocks, fetch.endian), blocks_x, blocks_y


def _rgb565(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.uint16, copy=False)
    result = np.empty((len(values), 4), dtype=np.uint8)
    result[:, 0] = ((values >> 11) & 31) * 255 // 31
    result[:, 1] = ((values >> 5) & 63) * 255 // 63
    result[:, 2] = (values & 31) * 255 // 31
    result[:, 3] = 255
    return result


def _color_block(blocks: np.ndarray, bc1: bool) -> np.ndarray:
    color0 = blocks[:, 0].astype(np.uint16) | (blocks[:, 1].astype(np.uint16) << 8)
    color1 = blocks[:, 2].astype(np.uint16) | (blocks[:, 3].astype(np.uint16) << 8)
    palette = np.empty((len(blocks), 4, 4), dtype=np.uint8)
    palette[:, 0] = _rgb565(color0)
    palette[:, 1] = _rgb565(color1)
    opaque = (color0 > color1) | (not bc1)
    for channel in range(3):
        first = palette[:, 0, channel].astype(np.uint16)
        second = palette[:, 1, channel].astype(np.uint16)
        palette[:, 2, channel] = np.where(
            opaque,
            (2 * first + second) // 3,
            (first + second) // 2,
        )
        palette[:, 3, channel] = np.where(
            opaque,
            (first + 2 * second) // 3,
            0,
        )
    palette[:, 2, 3] = 255
    palette[:, 3, 3] = np.where(opaque, 255, 0)
    bits = (
        blocks[:, 4].astype(np.uint32)
        | (blocks[:, 5].astype(np.uint32) << 8)
        | (blocks[:, 6].astype(np.uint32) << 16)
        | (blocks[:, 7].astype(np.uint32) << 24)
    )
    selectors = (bits[:, None] >> (np.arange(16, dtype=np.uint32) * 2)) & 3
    return palette[np.arange(len(blocks))[:, None], selectors]


def _alpha_block(blocks: np.ndarray) -> np.ndarray:
    alpha0 = blocks[:, 0].astype(np.uint16)
    alpha1 = blocks[:, 1].astype(np.uint16)
    palette = np.empty((len(blocks), 8), dtype=np.uint16)
    palette[:, 0] = alpha0
    palette[:, 1] = alpha1
    greater = alpha0 > alpha1
    for index in range(1, 7):
        seven_value = ((7 - index) * alpha0 + index * alpha1) // 7
        if index < 5:
            five_value = ((5 - index) * alpha0 + index * alpha1) // 5
        elif index == 5:
            five_value = np.zeros_like(alpha0)
        else:
            five_value = np.full_like(alpha0, 255)
        palette[:, index + 1] = np.where(greater, seven_value, five_value)
    bits = np.zeros(len(blocks), dtype=np.uint64)
    for byte_index in range(6):
        bits |= blocks[:, byte_index + 2].astype(np.uint64) << np.uint64(byte_index * 8)
    selectors = (bits[:, None] >> (np.arange(16, dtype=np.uint64) * 3)) & 7
    return palette[np.arange(len(blocks))[:, None], selectors].astype(np.uint8)


def _decode_compressed(blocks: np.ndarray, texture_format: int) -> np.ndarray:
    flat = blocks.reshape(-1, blocks.shape[-1])
    if texture_format == FORMAT_DXT1:
        return _color_block(flat, True)
    if texture_format in (FORMAT_DXT3, FORMAT_DXT5):
        rgba = _color_block(flat[:, 8:16], False)
        if texture_format == FORMAT_DXT3:
            alpha_bits = np.zeros(len(flat), dtype=np.uint64)
            for index in range(8):
                alpha_bits |= flat[:, index].astype(np.uint64) << np.uint64(index * 8)
            rgba[:, :, 3] = (
                (alpha_bits[:, None] >> (np.arange(16, dtype=np.uint64) * 4)) & 0xF
            ).astype(np.uint8) * 17
        else:
            rgba[:, :, 3] = _alpha_block(flat[:, :8])
        return rgba
    if texture_format == FORMAT_DXN:
        red = _alpha_block(flat[:, :8])
        green = _alpha_block(flat[:, 8:16])
        rgba = np.empty((len(flat), 16, 4), dtype=np.uint8)
        rgba[:, :, 0] = red
        rgba[:, :, 1] = green
        normal_x = red.astype(np.float32) / 127.5 - 1.0
        normal_y = green.astype(np.float32) / 127.5 - 1.0
        normal_z = np.sqrt(np.maximum(0.0, 1.0 - normal_x * normal_x - normal_y * normal_y))
        rgba[:, :, 2] = np.clip((normal_z + 1.0) * 127.5, 0, 255).astype(np.uint8)
        rgba[:, :, 3] = 255
        return rgba
    if texture_format == FORMAT_CTX1:
        palette = np.empty((len(flat), 4, 2), dtype=np.uint8)
        palette[:, 0] = flat[:, 0:2]
        palette[:, 1] = flat[:, 2:4]
        palette[:, 2] = (
            (2 * palette[:, 0].astype(np.uint16) + palette[:, 1].astype(np.uint16)) // 3
        ).astype(np.uint8)
        palette[:, 3] = (
            (palette[:, 0].astype(np.uint16) + 2 * palette[:, 1].astype(np.uint16)) // 3
        ).astype(np.uint8)
        bits = (
            flat[:, 4].astype(np.uint32)
            | (flat[:, 5].astype(np.uint32) << 8)
            | (flat[:, 6].astype(np.uint32) << 16)
            | (flat[:, 7].astype(np.uint32) << 24)
        )
        selectors = (bits[:, None] >> (np.arange(16, dtype=np.uint32) * 2)) & 3
        rg = palette[np.arange(len(flat))[:, None], selectors]
        rgba = np.empty((len(flat), 16, 4), dtype=np.uint8)
        rgba[:, :, :2] = rg
        normal_x = rg[:, :, 0].astype(np.float32) / 127.5 - 1.0
        normal_y = rg[:, :, 1].astype(np.float32) / 127.5 - 1.0
        normal_z = np.sqrt(np.maximum(0.0, 1.0 - normal_x * normal_x - normal_y * normal_y))
        rgba[:, :, 2] = np.clip((normal_z + 1.0) * 127.5, 0, 255).astype(np.uint8)
        rgba[:, :, 3] = 255
        return rgba
    if texture_format == FORMAT_DXT5A:
        alpha = _alpha_block(flat)
        return np.repeat(alpha[:, :, None], 4, axis=2)
    if texture_format in (FORMAT_DXT3A, FORMAT_DXT3A_1111):
        bits = np.zeros(len(flat), dtype=np.uint64)
        for index in range(8):
            bits |= flat[:, index].astype(np.uint64) << np.uint64(index * 8)
        alpha = (
            ((bits[:, None] >> (np.arange(16, dtype=np.uint64) * 4)) & 0xF).astype(np.uint8)
            * 17
        )
        return np.repeat(alpha[:, :, None], 4, axis=2)
    raise FormatError(f"unsupported compressed Xenos texture format {texture_format}")


def _decode_uncompressed(blocks: np.ndarray, texture_format: int) -> np.ndarray:
    flat = blocks.reshape(-1, blocks.shape[-1])
    rgba = np.empty((len(flat), 4), dtype=np.uint8)
    rgba[:, 3] = 255
    if texture_format == FORMAT_8:
        rgba[:, :3] = flat[:, :1]
    elif texture_format == FORMAT_88:
        rgba[:, 0:2] = flat[:, 0:2]
        rgba[:, 2] = 0
    elif texture_format == FORMAT_8888:
        rgba[:] = flat[:, :4]
    elif texture_format in (FORMAT_1555, FORMAT_565, FORMAT_4444):
        values = flat[:, 0].astype(np.uint16) | (flat[:, 1].astype(np.uint16) << 8)
        if texture_format == FORMAT_565:
            rgba[:] = _rgb565(values)
        elif texture_format == FORMAT_1555:
            rgba[:, 0] = ((values >> 10) & 31) * 255 // 31
            rgba[:, 1] = ((values >> 5) & 31) * 255 // 31
            rgba[:, 2] = (values & 31) * 255 // 31
            rgba[:, 3] = np.where(values & 0x8000, 255, 0)
        else:
            rgba[:, 0] = (values & 15) * 17
            rgba[:, 1] = ((values >> 4) & 15) * 17
            rgba[:, 2] = ((values >> 8) & 15) * 17
            rgba[:, 3] = ((values >> 12) & 15) * 17
    elif texture_format == FORMAT_RGBA16F:
        values = flat.view("<f2").reshape(-1, 4).astype(np.float32)
        rgba[:] = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        raise FormatError(f"unsupported uncompressed Xenos texture format {texture_format}")
    return rgba


def decode_texture(texture: XprTexture, face: int = 0) -> np.ndarray:
    if texture.tag_type == "TX3D":
        raise FormatError(
            f"XPR2 volume texture {texture.name!r} cannot be represented as a single Blender image"
        )
    if not 0 <= face < texture.face_count:
        raise FormatError(
            f"XPR2 texture {texture.name!r} has {texture.face_count} face(s), not face {face}"
        )
    fetch = texture.fetch
    block_size, _block_bytes = _FORMAT_INFO[fetch.format]
    blocks, blocks_x, blocks_y = _linear_blocks(texture, face)
    if block_size == 4:
        texels = _decode_compressed(blocks, fetch.format)
        rgba = (
            texels.reshape(blocks_y, blocks_x, 4, 4, 4)
            .transpose(0, 2, 1, 3, 4)
            .reshape(blocks_y * 4, blocks_x * 4, 4)
        )
    else:
        rgba = _decode_uncompressed(blocks, fetch.format).reshape(blocks_y, blocks_x, 4)
    rgba = rgba[: fetch.height, : fetch.width].copy()

    if fetch.swizzle != 0x688:
        original = rgba.copy()
        for channel in range(4):
            selector = (fetch.swizzle >> (channel * 3)) & 7
            if selector < 4:
                rgba[:, :, channel] = original[:, :, selector]
            elif selector == 5:
                rgba[:, :, channel] = 255
            else:
                rgba[:, :, channel] = 0
    return rgba
