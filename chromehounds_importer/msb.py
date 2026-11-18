from __future__ import annotations

from dataclasses import dataclass

from .binary import BinaryReader
from .errors import FormatError


@dataclass(frozen=True, slots=True)
class MsbModel:
    name: str
    model_type: int
    model_id: int


@dataclass(frozen=True, slots=True)
class MsbPart:
    name: str
    part_type: int
    part_id: int
    model_index: int
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]
    scale: tuple[float, float, float]
    layer_id: int


@dataclass(frozen=True, slots=True)
class MsbMap:
    version: int
    models: tuple[MsbModel, ...]
    parts: tuple[MsbPart, ...]


@dataclass(frozen=True, slots=True)
class _Param:
    version: int
    name: str
    records: tuple[int, ...]
    next_offset: int


def _read_param(reader: BinaryReader, offset: int) -> _Param:
    version, name_offset, count = reader.unpack_from(">III", offset, "MSB parameter header")
    if count < 1 or count > 1_000_000:
        raise FormatError(f"{reader.label}: invalid MSB parameter count {count} at 0x{offset:X}")
    offsets = reader.unpack_from(
        f">{count}I",
        offset + 12,
        "MSB parameter offsets",
    )
    name = reader.cstring_at(name_offset, encoding="cp932", what="MSB parameter name")
    return _Param(
        version=version,
        name=name,
        records=tuple(offsets[:-1]),
        next_offset=offsets[-1],
    )


def parse_msb(data: bytes, label: str = "<msb>") -> MsbMap:
    reader = BinaryReader(data, ">", label)
    params: dict[str, _Param] = {}
    offset = 0
    seen: set[int] = set()
    for _ in range(32):
        if offset in seen:
            raise FormatError(f"{label}: cyclic MSB parameter list at 0x{offset:X}")
        seen.add(offset)
        param = _read_param(reader, offset)
        params[param.name] = param
        if param.next_offset == 0:
            break
        if param.next_offset <= offset:
            raise FormatError(
                f"{label}: MSB parameter {param.name} points backward to 0x{param.next_offset:X}"
            )
        offset = param.next_offset
    else:
        raise FormatError(f"{label}: MSB has too many parameter tables")

    try:
        model_param = params["MODEL_PARAM_ST"]
        part_param = params["PARTS_PARAM_ST"]
    except KeyError as exc:
        raise FormatError(f"{label}: missing required MSB parameter {exc.args[0]}") from exc

    models: list[MsbModel] = []
    for index, record_offset in enumerate(model_param.records):
        reader.require(record_offset, 0x2C, f"MSB model {index}")
        name_offset, model_type, model_id = reader.unpack_from(
            ">Ihh", record_offset, f"MSB model {index}"
        )
        if model_type not in (0, 1, 2):
            raise FormatError(f"{label}: unsupported MSB model type {model_type}")
        models.append(
            MsbModel(
                name=reader.cstring_at(
                    record_offset + name_offset,
                    what=f"MSB model {index} name",
                ),
                model_type=model_type,
                model_id=model_id,
            )
        )

    parts: list[MsbPart] = []
    for index, record_offset in enumerate(part_param.records):
        reader.require(record_offset, 0x90, f"MSB part {index}")
        (
            name_offset,
            part_type,
            part_id,
            model_index,
            _unknown,
        ) = reader.unpack_from(">Ihhhh", record_offset, f"MSB part {index}")
        if part_type not in (0, 1, 2):
            raise FormatError(f"{label}: unsupported MSB part type {part_type}")
        position = reader.unpack_from(">3f", record_offset + 0x0C, "MSB part position")
        rotation = reader.unpack_from(">3f", record_offset + 0x18, "MSB part rotation")
        scale = reader.unpack_from(">3f", record_offset + 0x24, "MSB part scale")
        layer_id = reader.unpack_from(">h", record_offset + 0x30, "MSB part layer")[0]
        parts.append(
            MsbPart(
                name=reader.cstring_at(
                    record_offset + name_offset,
                    what=f"MSB part {index} name",
                ),
                part_type=part_type,
                part_id=part_id,
                model_index=model_index,
                position=position,
                rotation=rotation,
                scale=scale,
                layer_id=layer_id,
            )
        )
    return MsbMap(version=model_param.version, models=tuple(models), parts=tuple(parts))
