from __future__ import annotations

import struct
from dataclasses import dataclass

from .errors import FormatError


@dataclass(slots=True)
class BinaryReader:
    data: memoryview
    endian: str = ">"
    label: str = "<memory>"
    position: int = 0

    def __init__(
        self,
        data: bytes | bytearray | memoryview,
        endian: str = ">",
        label: str = "<memory>",
        position: int = 0,
    ) -> None:
        self.data = memoryview(data)
        self.endian = endian
        self.label = label
        self.position = position

    def __len__(self) -> int:
        return len(self.data)

    def require(self, offset: int, size: int, what: str = "data") -> None:
        if offset < 0 or size < 0 or offset + size > len(self.data):
            raise FormatError(
                f"{self.label}: {what} at 0x{offset:X}..0x{offset + size:X} "
                f"is outside the 0x{len(self.data):X}-byte file"
            )

    def seek(self, offset: int) -> None:
        self.require(offset, 0, "seek")
        self.position = offset

    def skip(self, size: int) -> None:
        self.seek(self.position + size)

    def unpack_from(self, fmt: str, offset: int, what: str = "field") -> tuple:
        full_fmt = fmt if fmt[:1] in "@=<>!" else self.endian + fmt
        size = struct.calcsize(full_fmt)
        self.require(offset, size, what)
        return struct.unpack_from(full_fmt, self.data, offset)

    def unpack(self, fmt: str, what: str = "field") -> tuple:
        full_fmt = fmt if fmt[:1] in "@=<>!" else self.endian + fmt
        size = struct.calcsize(full_fmt)
        values = self.unpack_from(full_fmt, self.position, what)
        self.position += size
        return values

    def u8(self) -> int:
        return self.unpack("B", "u8")[0]

    def i8(self) -> int:
        return self.unpack("b", "i8")[0]

    def u16(self) -> int:
        return self.unpack("H", "u16")[0]

    def i16(self) -> int:
        return self.unpack("h", "i16")[0]

    def u32(self) -> int:
        return self.unpack("I", "u32")[0]

    def i32(self) -> int:
        return self.unpack("i", "i32")[0]

    def f32(self) -> float:
        return self.unpack("f", "f32")[0]

    def bytes(self, size: int, what: str = "bytes") -> bytes:
        self.require(self.position, size, what)
        result = self.data[self.position : self.position + size].tobytes()
        self.position += size
        return result

    def bytes_at(self, offset: int, size: int, what: str = "bytes") -> bytes:
        self.require(offset, size, what)
        return self.data[offset : offset + size].tobytes()

    def fixed_string(
        self,
        size: int,
        encoding: str = "cp932",
        what: str = "string",
    ) -> str:
        raw = self.bytes(size, what).split(b"\0", 1)[0]
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as exc:
            raise FormatError(f"{self.label}: invalid {encoding} {what}") from exc

    def cstring_at(
        self,
        offset: int,
        encoding: str = "cp932",
        max_size: int | None = None,
        what: str = "string",
    ) -> str:
        self.require(offset, 1, what)
        end_limit = len(self.data) if max_size is None else min(len(self.data), offset + max_size)
        end = self.data[offset:end_limit].tobytes().find(b"\0")
        if end < 0:
            raise FormatError(f"{self.label}: unterminated {what} at 0x{offset:X}")
        raw = self.data[offset : offset + end].tobytes()
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as exc:
            raise FormatError(f"{self.label}: invalid {encoding} {what} at 0x{offset:X}") from exc

    def subreader(self, offset: int, size: int, label: str | None = None) -> "BinaryReader":
        self.require(offset, size, "subreader")
        return BinaryReader(
            self.data[offset : offset + size],
            endian=self.endian,
            label=label or self.label,
        )
