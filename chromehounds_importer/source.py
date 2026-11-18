from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path, PurePosixPath

from .errors import AssetNotFoundError
from .xdvdfs import XdvdfsImage


def normalise_asset_path(path: str) -> str:
    normalised = path.replace("\\", "/").lstrip("/")
    parts = PurePosixPath(normalised).parts
    if not normalised or any(part in {"", ".", ".."} for part in parts):
        raise AssetNotFoundError(f"Invalid virtual asset path: {path!r}")
    return "/".join(parts)


class AssetSource(ABC):
    @abstractmethod
    def read(self, path: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def exists(self, path: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def paths(self) -> tuple[str, ...]:
        raise NotImplementedError

    def find_by_basename(
        self,
        basename: str,
        suffix: str | None = None,
        preferred_prefix: str | None = None,
    ) -> tuple[str, ...]:
        folded = basename.casefold()
        candidates = [
            path
            for path in self.paths()
            if PurePosixPath(path).name.casefold() == folded
            and (suffix is None or path.casefold().endswith(suffix.casefold()))
        ]
        if preferred_prefix:
            prefix = normalise_asset_path(preferred_prefix).casefold().rstrip("/") + "/"
            candidates.sort(key=lambda path: (not path.casefold().startswith(prefix), len(path), path))
        else:
            candidates.sort(key=lambda path: (len(path), path))
        return tuple(candidates)


class DirectorySource(AssetSource):
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()
        self._paths: tuple[str, ...] | None = None

    def _resolve(self, path: str) -> Path:
        normalised = normalise_asset_path(path)
        candidate = self.root.joinpath(*PurePosixPath(normalised).parts).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise AssetNotFoundError(f"Asset path escapes the game root: {path!r}") from exc
        return candidate

    def read(self, path: str) -> bytes:
        candidate = self._resolve(path)
        if not candidate.is_file():
            raise AssetNotFoundError(f"{candidate} does not exist")
        return candidate.read_bytes()

    def exists(self, path: str) -> bool:
        return self._resolve(path).is_file()

    def paths(self) -> tuple[str, ...]:
        if self._paths is None:
            self._paths = tuple(
                child.relative_to(self.root).as_posix()
                for child in self.root.rglob("*")
                if child.is_file()
            )
        return self._paths


class IsoSource(AssetSource):
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.image = XdvdfsImage(path)
        self._paths: tuple[str, ...] | None = None

    def read(self, path: str) -> bytes:
        return self.image.read(path)

    def exists(self, path: str) -> bool:
        return self.image.exists(path)

    def paths(self) -> tuple[str, ...]:
        if self._paths is None:
            self._paths = tuple(entry.path for entry in self.image.entries())
        return self._paths


def open_asset_source(path: str | os.PathLike[str]) -> AssetSource:
    source = Path(path)
    if source.is_dir():
        return DirectorySource(source)
    if source.is_file() and source.suffix.casefold() in {".iso", ".xiso"}:
        return IsoSource(source)
    raise AssetNotFoundError(
        f"{source} is neither an extracted Chromehounds game directory nor an ISO image"
    )


def infer_game_root(asset_path: str | os.PathLike[str]) -> Path | None:
    path = Path(asset_path).resolve()
    for parent in (path.parent, *path.parents):
        if all((parent / name).exists() for name in ("chr", "parts", "mapdata")):
            return parent
    return None
