# Chromehounds Importer

Blender 5.2 extension for importing Chromehounds HOUND parts, models,
textures, animations, props, and instanced MSB levels from an extracted game
or directly from the retail ISO.

## Quick start

- Use **File > Import > Chromehounds Asset** for a local BND, MDL, SMD, M2A,
  XPR/XPC, MSB, or `.chbuild.json` file.
- Use the **Chromehounds** tab in the 3D View sidebar for direct ISO access.
  Choose an extracted game folder or ISO, enter a virtual path such as
  `parts/AX_AM001.bnd`, and select **Import from Game Source**.
- For a level, import its MSB, for example
  `mapdata/map/m01_001/m01_001.msb`. External props are collection-instanced.
- Set **Map Instance Limit** to a nonzero value for a quick level preview.

The extension creates bulk NumPy-backed meshes, Blender armatures and layered
Actions, internally packed generated images, and Principled materials. Xenos
texture tiling, endian conversion, swizzles, BC formats, DXN, and CTX1 are
decoded in memory. Imported image pixels remain embedded when a `.blend` file
is saved, moved, and reopened.

Opaque, alpha-blended, alpha-tested, and additive light-overlay materials are
mapped separately. Additive game overlays use transparent-plus-emission nodes
rather than treating their grayscale light masks as ordinary surface opacity.

## HOUND assemblies

Native garage/save build records are not decoded. Importer-owned
`.chbuild.json` files can assemble separate part bundles, parent them, and
attach them to named or numeric dummy sockets. The included
`chbuild.schema.json` describes version 1 of this format.

## Important limitations

- Proprietary M2A interpolation controls remain approximate; decoded
  translation, rotation, and scale keys use Blender Bezier interpolation.
- Game shaders and paint compositing are approximated with Principled nodes.
- HM2 collision, NVM navigation, particles, audio, scripts, and gameplay
  entities are not imported.
- Cube textures are exposed as six images.

See `THIRD_PARTY_NOTICES.md` for the MIT-licensed interoperability research
used by this extension.
