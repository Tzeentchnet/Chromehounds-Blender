from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

from .errors import AssetNotFoundError, FormatError

SECTOR_SIZE = 2048
VOLUME_MAGIC = b"MICROSOFT*XBOX*MEDIA"
PARTITION_OFFSETS = (0, 0x0FD90000, 0x02080000, 0x18300000)


@dataclass(frozen=True, slots=True)
class XdvdfsEntry:
    path: str
    offset: int
    size: int


def _find_partition(stream: BinaryIO, label: str) -> int:
    for base in PARTITION_OFFSETS:
        stream.seek(base + 32 * SECTOR_SIZE)
        if stream.read(len(VOLUME_MAGIC)) == VOLUME_MAGIC:
            return base
    raise FormatError(f"{label}: no supported XDVDFS game partition was found")


def _walk_directory(
    stream: BinaryIO,
    partition_base: int,
    sector: int,
    size: int,
    prefix: str = "",
) -> Iterator[XdvdfsEntry]:
    absolute = partition_base + sector * SECTOR_SIZE
    stream.seek(absolute)
    table = stream.read(size)
    if len(table) != size:
        raise FormatError(
            f"XDVDFS directory {prefix or '/'} is truncated: got {len(table)}, expected {size}"
        )

    pending = [0]
    seen: set[int] = set()
    while pending:
        node_words = pending.pop()
        node_offset = node_words * 4
        if node_offset in seen:
            continue
        seen.add(node_offset)
        if node_offset + 14 > len(table):
            raise FormatError(
                f"XDVDFS directory {prefix or '/'} has an out-of-range node at 0x{node_offset:X}"
            )
        left, right, start, length, attributes, name_length = struct.unpack_from(
            "<HHIIBB", table, node_offset
        )
        if left == 0xFFFF:
            continue
        name_end = node_offset + 14 + name_length
        if name_end > len(table):
            raise FormatError(f"XDVDFS directory {prefix or '/'} has a truncated file name")
        name = table[node_offset + 14 : name_end].decode("latin-1")
        path = f"{prefix}{name}"
        if attributes & 0x10:
            if length:
                yield from _walk_directory(
                    stream,
                    partition_base,
                    start,
                    length,
                    f"{path}/",
                )
        else:
            yield XdvdfsEntry(
                path=path,
                offset=partition_base + start * SECTOR_SIZE,
                size=length,
            )
        if left:
            pending.append(left)
        if right:
            pending.append(right)


class XdvdfsImage:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._entries: dict[str, XdvdfsEntry] | None = None

    def _load_index(self) -> dict[str, XdvdfsEntry]:
        if self._entries is not None:
            return self._entries
        with self.path.open("rb") as stream:
            base = _find_partition(stream, str(self.path))
            descriptor = base + 32 * SECTOR_SIZE
            stream.seek(descriptor + len(VOLUME_MAGIC))
            raw = stream.read(8)
            if len(raw) != 8:
                raise FormatError(f"{self.path}: truncated XDVDFS volume descriptor")
            root_sector, root_size = struct.unpack("<II", raw)
            entries = tuple(_walk_directory(stream, base, root_sector, root_size))
        self._entries = {entry.path.replace("\\", "/").casefold(): entry for entry in entries}
        return self._entries

    def entries(self) -> tuple[XdvdfsEntry, ...]:
        return tuple(self._load_index().values())

    def exists(self, path: str) -> bool:
        return _normalise_path(path).casefold() in self._load_index()

    def read(self, path: str) -> bytes:
        normalised = _normalise_path(path)
        entry = self._load_index().get(normalised.casefold())
        if entry is None:
            raise AssetNotFoundError(f"{self.path}: {normalised!r} is not present in the ISO")
        with self.path.open("rb") as stream:
            stream.seek(entry.offset)
            data = stream.read(entry.size)
        if len(data) != entry.size:
            raise FormatError(
                f"{self.path}: {normalised!r} is truncated: got {len(data)}, expected {entry.size}"
            )
        return data


def _normalise_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("/")
