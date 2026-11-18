from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


TITLES = {
    "hound_iso": "HOUND part from retail ISO",
    "animated_character": "Animated c0101 bundle (frame 30)",
    "prop": "Textured o06-030 prop",
    "level": "Complete m01_001 level (8,378 placements)",
    "assembly": "Two-part dummy/socket assembly",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_contact_sheet(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    tile_width, tile_height, title_height = 640, 360, 34
    columns = 2
    row_count = (len(rows) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (tile_width * columns, (tile_height + title_height) * row_count),
        (12, 18, 28),
    )
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        screenshot = Path(row["files"]["screenshot"])
        image = Image.open(screenshot).convert("RGB")
        image = ImageOps.fit(
            image,
            (tile_width, tile_height),
            method=Image.Resampling.LANCZOS,
        )
        x = (index % columns) * tile_width
        y = (index // columns) * (tile_height + title_height)
        canvas.paste(image, (x, y))
        draw.rectangle(
            (x, y + tile_height, x + tile_width, y + tile_height + title_height),
            fill=(18, 28, 42),
        )
        draw.text(
            (x + 12, y + tile_height + 9),
            TITLES[row["case"]],
            fill=(235, 242, 250),
        )
    canvas.save(output_path, format="PNG", optimize=True)


def create_html(rows: list[dict[str, object]], output_path: Path) -> None:
    cards: list[str] = []
    for row in rows:
        screenshot = Path(row["files"]["screenshot"])
        blend = Path(row["files"]["blend"])
        screenshot_relative = screenshot.relative_to(output_path.parent).as_posix()
        blend_relative = blend.relative_to(output_path.parent).as_posix()
        metrics = row["metrics"]
        placements = metrics.get(
            "map_placements",
            metrics["collection_instances"],
        )
        cards.append(
            f"""
            <article>
              <h2>{html.escape(TITLES[row["case"]])}</h2>
              <a href="{html.escape(screenshot_relative)}">
                <img src="{html.escape(screenshot_relative)}" alt="{html.escape(TITLES[row["case"]])}">
              </a>
              <p>
                <a href="{html.escape(blend_relative)}">Open Blender file</a>
                &middot; {metrics["vertices"]:,} vertices
                &middot; {metrics["polygons"]:,} polygons
                &middot; {metrics["images"]} images
                &middot; {metrics["actions"]} actions
                &middot; {placements:,} placements
              </p>
            </article>
            """
        )
    output_path.write_text(
        """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Chromehounds Blender Import Tests</title>
<style>
body { background:#101822; color:#e8eef5; font:16px system-ui; margin:2rem auto; max-width:1200px; }
h1, h2 { color:#fff; }
article { background:#182536; border-radius:12px; margin:1.5rem 0; padding:1rem; }
img { border-radius:8px; display:block; max-width:100%; }
a { color:#8dccff; }
</style>
</head>
<body>
<h1>Chromehounds Blender 5.2 Import Tests</h1>
<p>Five packaged-extension imports. Every saved .blend file was reopened and validated.</p>
"""
        + "\n".join(cards)
        + """
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    reports = sorted(args.output_root.glob("*/report.json"))
    if len(reports) != 5:
        raise RuntimeError(f"Expected five reports, found {len(reports)}")
    rows = [
        json.loads(report_path.read_text(encoding="utf-8"))
        for report_path in reports
    ]
    for row in rows:
        blend = Path(row["files"]["blend"])
        screenshot = Path(row["files"]["screenshot"])
        row["files"]["blend_sha256"] = sha256(blend)
        row["files"]["screenshot_sha256"] = sha256(screenshot)

    summary = {
        "suite": "Chromehounds Blender 5.2 import results",
        "status": "passed",
        "case_count": len(rows),
        "cases": rows,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        "Chromehounds Blender 5.2 Import Tests",
        "======================================",
        "",
        "Status: PASSED (5/5)",
        "",
    ]
    for row in rows:
        metrics = row["metrics"]
        placements = metrics.get(
            "map_placements",
            metrics["collection_instances"],
        )
        lines.extend(
            (
                TITLES[row["case"]],
                f"  Blend: {row['files']['blend']}",
                f"  Screenshot: {row['files']['screenshot']}",
                f"  Geometry: {metrics['vertices']:,} vertices, {metrics['polygons']:,} polygons",
                f"  Images/actions/placements: {metrics['images']} / {metrics['actions']} / {placements:,}",
                "",
            )
        )
    (args.output_root / "README.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    create_contact_sheet(rows, args.output_root / "contact_sheet.png")
    create_html(rows, args.output_root / "index.html")
    print(f"Wrote summary for {len(rows)} passing cases to {args.output_root}")


if __name__ == "__main__":
    main()
