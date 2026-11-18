from __future__ import annotations

import zlib
from dataclasses import dataclass

from .binary import BinaryReader
from .errors import FormatError


@dataclass(frozen=True, slots=True)
class BndEntry:
    name: str
    file_id: int
    flags: int
    packed_size: int
    unpacked_size: int
    data_offset: int
    data: bytes


@dataclass(frozen=True, slots=True)
class BndArchive:
    version: str
    flags: int
    entries: tuple[BndEntry, ...]

    def by_name(self) -> dict[str, BndEntry]:
        return {entry.name.casefold(): entry for entry in self.entries}


def parse_bnd(data: bytes, label: str = "<bnd>") -> BndArchive:
    reader = BinaryReader(data, ">", label)
    if len(reader) < 32 or reader.bytes_at(0, 4) != b"BND3":
        raise FormatError(f"{label}: expected a BND3 archive")

    try:
        version = reader.bytes_at(4, 8, "BND version").decode("ascii")
    except UnicodeDecodeError as exc:
        raise FormatError(f"{label}: invalid BND version") from exc

    flags = reader.unpack_from(">I", 12, "BND flags")[0]
    file_count = reader.unpack_from(">I", 16, "BND file count")[0]
    if file_count > 1_000_000:
        raise FormatError(f"{label}: unreasonable BND file count {file_count}")

    format_byte = data[12]
    if format_byte == 0xE4:
        entry_size = 24
        compressed_variant = True
    elif format_byte in (0xE0, 0xF0):
        entry_size = 20
        compressed_variant = False
    else:
        raise FormatError(f"{label}: unsupported BND3 flags 0x{flags:08X}")

    table_end = 32 + file_count * entry_size
    reader.require(32, file_count * entry_size, "BND entry table")
    entries: list[BndEntry] = []
    for index in range(file_count):
        offset = 32 + index * entry_size
        values = reader.unpack_from(
            ">6I" if compressed_variant else ">5I",
            offset,
            f"BND entry {index}",
        )
        entry_flags, packed_size, data_offset, file_id, name_offset = values[:5]
        unpacked_size = values[5] if compressed_variant else packed_size
        if name_offset < table_end:
            raise FormatError(
                f"{label}: entry {index} name offset 0x{name_offset:X} overlaps its table"
            )
        name = reader.cstring_at(name_offset, what=f"BND entry {index} name")
        packed = reader.bytes_at(data_offset, packed_size, f"BND entry {index} payload")

        if compressed_variant and unpacked_size != packed_size:
            try:
                payload = zlib.decompress(packed)
            except zlib.error as exc:
                raise FormatError(
                    f"{label}: entry {name!r} is marked compressed but zlib rejected it"
                ) from exc
            if len(payload) != unpacked_size:
                raise FormatError(
                    f"{label}: entry {name!r} expanded to {len(payload)} bytes, "
                    f"expected {unpacked_size}"
                )
        else:
            payload = packed

        entries.append(
            BndEntry(
                name=name,
                file_id=file_id,
                flags=entry_flags,
                packed_size=packed_size,
                unpacked_size=unpacked_size,
                data_offset=data_offset,
                data=payload,
            )
        )

    return BndArchive(version=version, flags=flags, entries=tuple(entries))
