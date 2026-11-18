from __future__ import annotations

from pathlib import Path

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import FileHandler, Operator, Panel, PropertyGroup
from bpy_extras.io_utils import ImportHelper

from .errors import ChromehoundsError
from .importer import ImportResult, ImportSettings, import_asset_data
from .source import DirectorySource, infer_game_root, normalise_asset_path, open_asset_source

_FILTER_GLOB = (
    "*.bnd;*.mdl;*.smd;*.m2a;*.xpr;*.xpc;*.msb;*.chbuild.json"
)

_MAP_PLACEMENT_ITEMS = (
    (
        "LINKED_OBJECTS",
        "Selectable Linked Objects",
        "Create selectable objects that share mesh and material datablocks",
    ),
    (
        "COLLECTION_INSTANCES",
        "Collection Instances",
        "Create lightweight instance empties without directly selectable materials",
    ),
)


class CHROME_PG_import_settings(PropertyGroup):
    source_path: StringProperty(
        name="Game Source",
        description="Extracted Chromehounds game folder or retail ISO image",
        subtype="FILE_PATH",
    )
    asset_path: StringProperty(
        name="Asset Path",
        description="Virtual game path, for example parts/AX_AM001.bnd",
        default="parts/AX_AM001.bnd",
    )
    unit_scale: FloatProperty(
        name="Unit Scale",
        description="Blender units per Chromehounds game unit",
        default=0.01,
        min=0.000001,
        soft_max=1.0,
        precision=5,
    )
    animation_fps: IntProperty(
        name="Animation FPS",
        description="Scene frame rate used for imported M2A clips",
        default=30,
        min=1,
        max=240,
    )
    import_lods: BoolProperty(name="Import LODs", default=False)
    import_shadows: BoolProperty(name="Import Shadow Meshes", default=False)
    import_animations: BoolProperty(name="Import Animations", default=True)
    import_textures: BoolProperty(name="Import Textures", default=True)
    import_external_assets: BoolProperty(
        name="Resolve Map Props",
        description="Load object and character bundles referenced by MSB maps",
        default=True,
    )
    create_armatures: BoolProperty(name="Create Armatures", default=True)
    map_placement_mode: EnumProperty(
        name="Map Placements",
        description="Choose editable linked objects or lightweight collection instances",
        items=_MAP_PLACEMENT_ITEMS,
        default="LINKED_OBJECTS",
    )
    max_map_instances: IntProperty(
        name="Map Instance Limit",
        description="Maximum placements to import; zero imports the complete map",
        default=0,
        min=0,
    )


def _settings_from_properties(owner) -> ImportSettings:
    return ImportSettings(
        unit_scale=owner.unit_scale,
        animation_fps=owner.animation_fps,
        import_lods=owner.import_lods,
        import_shadows=owner.import_shadows,
        import_animations=owner.import_animations,
        import_textures=owner.import_textures,
        import_external_assets=owner.import_external_assets,
        create_armatures=owner.create_armatures,
        max_map_instances=owner.max_map_instances,
        map_placement_mode=owner.map_placement_mode,
    )


def _report_result(operator: Operator, result: ImportResult) -> None:
    operator.report(
        {"INFO"},
        "Imported "
        f"{result.models} model(s), {result.meshes} mesh(es), "
        f"{result.images} image(s), {result.actions} action(s), "
        f"{result.instances} map placement(s)",
    )
    for warning in result.warnings[:10]:
        operator.report({"WARNING"}, warning)
    if len(result.warnings) > 10:
        operator.report(
            {"WARNING"},
            f"{len(result.warnings) - 10} additional warnings were omitted",
        )


def _draw_options(layout: bpy.types.UILayout, owner) -> None:
    layout.prop(owner, "unit_scale")
    layout.prop(owner, "animation_fps")
    row = layout.row(align=True)
    row.prop(owner, "import_textures")
    row.prop(owner, "create_armatures")
    row = layout.row(align=True)
    row.prop(owner, "import_lods")
    row.prop(owner, "import_shadows")
    layout.prop(owner, "import_animations")
    layout.prop(owner, "import_external_assets")
    layout.prop(owner, "map_placement_mode")
    layout.prop(owner, "max_map_instances")


class CHROME_IMPORT_OT_asset(Operator, ImportHelper):
    bl_idname = "chromehounds.import_asset"
    bl_label = "Import Chromehounds Asset"
    bl_description = "Import a Chromehounds bundle, model, texture, animation, map, or build"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ""
    filter_glob: StringProperty(default=_FILTER_GLOB, options={"HIDDEN"})
    unit_scale: FloatProperty(default=0.01, min=0.000001, soft_max=1.0)
    animation_fps: IntProperty(name="Animation FPS", default=30, min=1, max=240)
    import_lods: BoolProperty(name="Import LODs", default=False)
    import_shadows: BoolProperty(name="Import Shadow Meshes", default=False)
    import_animations: BoolProperty(name="Import Animations", default=True)
    import_textures: BoolProperty(name="Import Textures", default=True)
    import_external_assets: BoolProperty(name="Resolve Map Props", default=True)
    create_armatures: BoolProperty(name="Create Armatures", default=True)
    map_placement_mode: EnumProperty(
        name="Map Placements",
        description="Choose editable linked objects or lightweight collection instances",
        items=_MAP_PLACEMENT_ITEMS,
        default="LINKED_OBJECTS",
    )
    max_map_instances: IntProperty(name="Map Instance Limit", default=0, min=0)

    def draw(self, _context: bpy.types.Context) -> None:
        _draw_options(self.layout, self)

    def execute(self, context: bpy.types.Context) -> set[str]:
        path = Path(bpy.path.abspath(self.filepath))
        try:
            data = path.read_bytes()
            root = infer_game_root(path)
            source = DirectorySource(root) if root is not None else None
            virtual_path = (
                path.resolve().relative_to(root.resolve()).as_posix()
                if root is not None
                else None
            )
            if source is None:
                configured = context.scene.chromehounds_import.source_path
                if configured:
                    source = open_asset_source(Path(bpy.path.abspath(configured)))
            result = import_asset_data(
                data,
                str(path),
                context,
                _settings_from_properties(self),
                source=source,
                virtual_path=virtual_path,
                local_path=path,
            )
        except (ChromehoundsError, OSError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _report_result(self, result)
        return {"FINISHED"}


class CHROME_IMPORT_OT_from_source(Operator):
    bl_idname = "chromehounds.import_from_source"
    bl_label = "Import from Game Source"
    bl_description = "Read a virtual asset directly from an extracted game or retail ISO"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        props = context.scene.chromehounds_import
        source_path = Path(bpy.path.abspath(props.source_path))
        virtual_path = normalise_asset_path(props.asset_path)
        try:
            source = open_asset_source(source_path)
            data = source.read(virtual_path)
            result = import_asset_data(
                data,
                virtual_path,
                context,
                _settings_from_properties(props),
                source=source,
                virtual_path=virtual_path,
            )
        except (ChromehoundsError, OSError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        _report_result(self, result)
        return {"FINISHED"}


class CHROME_OT_select_game_folder(Operator):
    bl_idname = "chromehounds.select_game_folder"
    bl_label = "Select Extracted Game Folder"
    bl_description = "Choose the root of an extracted Chromehounds game"

    directory: StringProperty(subtype="DIR_PATH")

    def invoke(
        self,
        context: bpy.types.Context,
        _event: bpy.types.Event,
    ) -> set[str]:
        current = context.scene.chromehounds_import.source_path
        if current and Path(bpy.path.abspath(current)).is_dir():
            self.directory = current
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        context.scene.chromehounds_import.source_path = self.directory
        return {"FINISHED"}


class CHROME_OT_select_iso(Operator, ImportHelper):
    bl_idname = "chromehounds.select_iso"
    bl_label = "Select Chromehounds ISO"
    bl_description = "Choose a retail Chromehounds ISO or XISO image"

    filename_ext = ".iso"
    filter_glob: StringProperty(default="*.iso;*.xiso", options={"HIDDEN"})

    def execute(self, context: bpy.types.Context) -> set[str]:
        context.scene.chromehounds_import.source_path = self.filepath
        return {"FINISHED"}


class CHROME_PT_import(Panel):
    bl_label = "Chromehounds Importer"
    bl_idname = "CHROME_PT_import"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Chromehounds"

    def draw(self, context: bpy.types.Context) -> None:
        props = context.scene.chromehounds_import
        layout = self.layout
        layout.prop(props, "source_path")
        source_row = layout.row(align=True)
        source_row.operator(
            CHROME_OT_select_game_folder.bl_idname,
            text="Game Folder",
            icon="FILE_FOLDER",
        )
        source_row.operator(
            CHROME_OT_select_iso.bl_idname,
            text="ISO",
            icon="DISC",
        )
        layout.prop(props, "asset_path")
        layout.operator(CHROME_IMPORT_OT_from_source.bl_idname, icon="IMPORT")
        options = layout.box()
        options.label(text="Import Options")
        _draw_options(options, props)
        layout.separator()
        layout.operator(CHROME_IMPORT_OT_asset.bl_idname, text="Import Local Asset")


class CHROME_FH_asset(FileHandler):
    bl_idname = "CHROME_FH_asset"
    bl_label = "Chromehounds Asset"
    bl_import_operator = CHROME_IMPORT_OT_asset.bl_idname
    bl_file_extensions = ".bnd;.mdl;.smd;.m2a;.xpr;.xpc;.msb;.json"

    @classmethod
    def poll_drop(cls, context: bpy.types.Context) -> bool:
        return context.area is not None and context.area.type in {"VIEW_3D", "OUTLINER"}


def _menu_import(self, _context: bpy.types.Context) -> None:
    self.layout.operator(
        CHROME_IMPORT_OT_asset.bl_idname,
        text="Chromehounds Asset (.bnd/.mdl/.xpr/.msb)",
    )


CLASSES = (
    CHROME_PG_import_settings,
    CHROME_IMPORT_OT_asset,
    CHROME_IMPORT_OT_from_source,
    CHROME_OT_select_game_folder,
    CHROME_OT_select_iso,
    CHROME_PT_import,
    CHROME_FH_asset,
)


def register() -> None:
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.chromehounds_import = bpy.props.PointerProperty(
        type=CHROME_PG_import_settings
    )
    bpy.types.TOPBAR_MT_file_import.append(_menu_import)


def unregister() -> None:
    bpy.types.TOPBAR_MT_file_import.remove(_menu_import)
    del bpy.types.Scene.chromehounds_import
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
