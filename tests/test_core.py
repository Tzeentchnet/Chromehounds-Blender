from __future__ import annotations

import hashlib
import os
import struct
import unittest
import zlib
from pathlib import Path

import numpy as np

from chromehounds_importer.bnd import parse_bnd
from chromehounds_importer.errors import AssetNotFoundError, FormatError
from chromehounds_importer.m2a import parse_m2a
from chromehounds_importer.model import parse_mdl4, triangulate_strip
from chromehounds_importer.source import DirectorySource, IsoSource
from chromehounds_importer.texture import decode_texture, parse_xpr


def _make_bnd(payload: bytes, *, compressed: bool) -> bytes:
    name = b"fixture.bin\0"
    packed = zlib.compress(payload) if compressed else payload
    entry_size = 24 if compressed else 20
    name_offset = 32 + entry_size
    data_offset = name_offset + len(name)
    result = bytearray(data_offset + len(packed))
    result[0:12] = b"BND305I29V48"
    struct.pack_into(">I", result, 12, 0xE4000000 if compressed else 0xE0000000)
    struct.pack_into(">I", result, 16, 1)
    values = (0, len(packed), data_offset, 7, name_offset)
    struct.pack_into(">5I", result, 32, *values)
    if compressed:
        struct.pack_into(">I", result, 52, len(payload))
    result[name_offset : name_offset + len(name)] = name
    result[data_offset:] = packed
    return bytes(result)


class CoreParserTests(unittest.TestCase):
    def test_uncompressed_bnd(self) -> None:
        archive = parse_bnd(_make_bnd(b"Chromehounds", compressed=False))
        self.assertEqual(archive.entries[0].name, "fixture.bin")
        self.assertEqual(archive.entries[0].file_id, 7)
        self.assertEqual(archive.entries[0].data, b"Chromehounds")

    def test_compressed_bnd(self) -> None:
        archive = parse_bnd(_make_bnd(b"A" * 4096, compressed=True))
        self.assertEqual(archive.entries[0].data, b"A" * 4096)
        self.assertLess(
            archive.entries[0].packed_size,
            archive.entries[0].unpacked_size,
        )

    def test_bnd_rejects_truncated_payload(self) -> None:
        fixture = _make_bnd(b"fixture", compressed=False)
        with self.assertRaises(FormatError):
            parse_bnd(fixture[:-1])

    def test_triangle_strip_restart_and_winding(self) -> None:
        indices = np.asarray([0, 1, 2, 3, 0xFFFF, 4, 5, 6], dtype=np.uint16)
        triangles = triangulate_strip(indices)
        np.testing.assert_array_equal(
            triangles,
            np.asarray(((0, 1, 2), (3, 2, 1), (4, 5, 6)), dtype=np.int32),
        )
        np.testing.assert_array_equal(
            triangulate_strip(indices, flip_winding=True),
            np.asarray(((2, 1, 0), (1, 2, 3), (6, 5, 4)), dtype=np.int32),
        )

    def test_directory_source_blocks_parent_traversal(self) -> None:
        source = DirectorySource(Path(__file__).parent)
        with self.assertRaises(AssetNotFoundError):
            source.read("../test_core.py")


@unittest.skipUnless(
    os.environ.get("CHROMEHOUNDS_GAME_ROOT"),
    "set CHROMEHOUNDS_GAME_ROOT to run retail-asset tests",
)
class RetailAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(os.environ["CHROMEHOUNDS_GAME_ROOT"])

    def test_hound_part_model_and_textures(self) -> None:
        path = self.root / "parts" / "AX_AM001.bnd"
        archive = parse_bnd(path.read_bytes(), str(path))
        model_entry = next(entry for entry in archive.entries if entry.name == "AX_AM001.mdl")
        model = parse_mdl4(model_entry.data, model_entry.name)
        self.assertEqual(model.version, 0x40002)
        self.assertEqual(len(model.nodes), 2)
        self.assertEqual(len(model.meshes), 1)
        self.assertEqual(len(model.meshes[0].positions), 748)
        self.assertEqual(len(model.meshes[0].triangles()), 424)
        triangles = model.meshes[0].triangles(flip_winding=True)
        positions = model.meshes[0].positions
        normals = model.meshes[0].normals
        face_normals = np.cross(
            positions[triangles[:, 1]] - positions[triangles[:, 0]],
            positions[triangles[:, 2]] - positions[triangles[:, 0]],
        )
        alignment = (face_normals * normals[triangles].sum(axis=1)).sum(axis=1)
        self.assertGreater(np.count_nonzero(alignment > 0), len(alignment) * 0.99)

        xpr_entry = next(entry for entry in archive.entries if entry.name == "AX_AM001.xpr")
        package = parse_xpr(xpr_entry.data, xpr_entry.name)
        expected = {
            "AX_AM001": "c7afe62ad29c797a814347241d6e9e043bde5be448be7745ea70f320f4a04c0d",
            "AX_AM001_b": "308b216ccc2fbf2bed7f77bf33660e4bf04452e220c35fd420e8c4b9a92be549",
        }
        for texture in package.textures:
            rgba = decode_texture(texture)
            self.assertEqual(rgba.shape, (256, 256, 4))
            self.assertEqual(
                hashlib.sha256(rgba.tobytes()).hexdigest(),
                expected[texture.name],
            )

    def test_multi_mesh_map_models_advance_correctly(self) -> None:
        path = self.root / "mapdata" / "map" / "m01_001" / "m01_001.bnd"
        archive = parse_bnd(path.read_bytes(), str(path))
        entry = next(item for item in archive.entries if item.name == "m0000.mdl")
        model = parse_mdl4(entry.data, entry.name)
        self.assertEqual(len(model.meshes), 3)
        self.assertEqual([mesh.material_index for mesh in model.meshes], [0, 1, 2])

    def test_compact_xpr_descriptor(self) -> None:
        path = self.root / "mapdata" / "obj" / "o06-030" / "o06_030.xpr"
        package = parse_xpr(path.read_bytes(), str(path))
        self.assertEqual(len(package.textures), 2)
        self.assertEqual(decode_texture(package.textures[0]).shape, (512, 512, 4))

    def test_m2a_300_variant(self) -> None:
        path = self.root / "chr" / "c1000.bnd"
        archive = parse_bnd(path.read_bytes(), str(path))
        entry = next(item for item in archive.entries if item.name == "c1000.m2a")
        animation = parse_m2a(entry.data, entry.name)
        self.assertEqual(animation.version, 0x300)
        self.assertEqual(animation.node_count, 63)
        self.assertEqual(len(animation.clips), 3)

    def test_f0_bnd_and_cp932_names(self) -> None:
        script_path = self.root / "AI" / "Script" / "AI_Script.bnd"
        self.assertEqual(len(parse_bnd(script_path.read_bytes()).entries), 75)

        object_path = self.root / "mapdata" / "obj" / "o00-001" / "o00-001.bnd"
        archive = parse_bnd(object_path.read_bytes(), str(object_path))
        model_entry = next(item for item in archive.entries if item.name == "o00-001.mdl")
        self.assertGreater(len(parse_mdl4(model_entry.data).nodes), 0)

        character_path = self.root / "chr" / "c7001.bnd"
        character = parse_bnd(character_path.read_bytes(), str(character_path))
        xpr_entry = next(item for item in character.entries if item.data[:4] == b"XPR2")
        self.assertGreater(len(parse_xpr(xpr_entry.data).textures), 0)


@unittest.skipUnless(
    os.environ.get("CHROMEHOUNDS_ISO") and os.environ.get("CHROMEHOUNDS_GAME_ROOT"),
    "set CHROMEHOUNDS_ISO and CHROMEHOUNDS_GAME_ROOT to run direct-ISO tests",
)
class RetailIsoTests(unittest.TestCase):
    def test_iso_asset_matches_extracted_file(self) -> None:
        source = IsoSource(os.environ["CHROMEHOUNDS_ISO"])
        actual = source.read("parts/AX_AM001.bnd")
        expected = (
            Path(os.environ["CHROMEHOUNDS_GAME_ROOT"]) / "parts" / "AX_AM001.bnd"
        ).read_bytes()
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
