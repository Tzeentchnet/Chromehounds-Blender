from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .binary import BinaryReader
from .errors import FormatError


@dataclass(frozen=True, slots=True)
class M2aKey:
    frame: int
    span: int
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float]
    scale: tuple[float, float, float]
    translation_controls: tuple[int, int]
    rotation_controls: tuple[int, int]


@dataclass(frozen=True, slots=True)
class M2aTrack:
    keys: tuple[M2aKey, ...]


@dataclass(frozen=True, slots=True)
class M2aClip:
    clip_id: int
    duration: int
    tracks: tuple[M2aTrack, ...]


@dataclass(frozen=True, slots=True)
class M2aAnimation:
    version: int
    node_count: int
    clips: tuple[M2aClip, ...]


def parse_m2a(data: bytes, label: str = "<m2a>") -> M2aAnimation:
    reader = BinaryReader(data, ">", label)
    if len(reader) < 0x20 or reader.bytes_at(0, 4) != b"M2Af":
        raise FormatError(f"{label}: expected an M2Af animation")
    reader.seek(4)
    declared_size = reader.u32()
    version = reader.u32()
    clip_count = reader.u32()
    node_count = reader.u32()
    pool_count = reader.u32()
    reader.skip(8)
    if declared_size > len(data):
        raise FormatError(
            f"{label}: declared size 0x{declared_size:X} exceeds file size 0x{len(data):X}"
        )
    if version not in (0x300, 0x301):
        raise FormatError(f"{label}: unsupported M2A version 0x{version:X}")
    if clip_count > 10_000 or node_count > 100_000 or pool_count > 1_000_000:
        raise FormatError(f"{label}: unreasonable M2A counts")

    directories = tuple(
        reader.unpack_from(">HHI", 0x20 + index * 8, f"M2A clip {index} directory")
        for index in range(clip_count)
    )
    pool_offset = 0x20 + clip_count * 8
    reader.require(pool_offset, pool_count * 6, "M2A value pool")
    pool_raw = np.frombuffer(
        reader.data[pool_offset : pool_offset + pool_count * 6],
        dtype=">u2",
    ).reshape(pool_count, 3)
    translation_pool = pool_raw.view(">f2").astype(np.float32)
    rotation_pool = (
        pool_raw.view(">i2").astype(np.float32) * np.float32(math.pi / 32768.0)
    )
    scale_pool = pool_raw.view(">i2").astype(np.float32) / np.float32(1000.0)

    clips: list[M2aClip] = []
    for clip_index, (clip_id, duration, track_table_offset) in enumerate(directories):
        reader.require(
            track_table_offset,
            node_count * 0x20,
            f"M2A clip {clip_index} track table",
        )
        tracks: list[M2aTrack] = []
        for node_index in range(node_count):
            track_offset = track_table_offset + node_index * 0x20
            key_count = reader.unpack_from(
                ">H", track_offset, f"M2A clip {clip_index} node {node_index} key count"
            )[0]
            key_offset = reader.unpack_from(
                ">I", track_offset + 8, f"M2A clip {clip_index} node {node_index} keys"
            )[0]
            if key_count == 0:
                tracks.append(M2aTrack(keys=()))
                continue
            reader.require(
                key_offset,
                key_count * 0x20,
                f"M2A clip {clip_index} node {node_index} key data",
            )
            keys: list[M2aKey] = []
            for key_index in range(key_count):
                values = reader.unpack_from(
                    ">HH7I",
                    key_offset + key_index * 0x20,
                    f"M2A clip {clip_index} node {node_index} key {key_index}",
                )
                frame, span = values[:2]
                refs = values[2:]
                if any(ref >= pool_count for ref in refs):
                    raise FormatError(
                        f"{label}: M2A clip {clip_index} node {node_index} key {key_index} "
                        f"references a value outside the {pool_count}-entry pool"
                    )
                translation = translation_pool[refs[0]]
                rotation = rotation_pool[refs[3]]
                scale = scale_pool[refs[6]]
                keys.append(
                    M2aKey(
                        frame=frame,
                        span=span,
                        translation=(
                            float(translation[0]),
                            float(-translation[1]),
                            float(translation[2]),
                        ),
                        rotation=tuple(float(value) for value in rotation),
                        scale=tuple(float(value) for value in scale),
                        translation_controls=(refs[1], refs[2]),
                        rotation_controls=(refs[4], refs[5]),
                    )
                )
            tracks.append(M2aTrack(keys=tuple(keys)))
        clips.append(M2aClip(clip_id=clip_id, duration=duration, tracks=tuple(tracks)))
    return M2aAnimation(version=version, node_count=node_count, clips=tuple(clips))
