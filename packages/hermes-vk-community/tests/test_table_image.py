from pathlib import Path

from PIL import Image

from hermes_vk_community.renderer import RenderedTableSegment
from hermes_vk_community.table_image import render_table_jpegs


def test_table_is_rendered_as_readable_rgb_jpeg(tmp_path: Path) -> None:
    table = RenderedTableSegment(
        headers=("Поле", "Значение"),
        rows=(("План", "Pro"), ("Срок", "30 дней")),
    )
    paths = render_table_jpegs(table, tmp_path)
    assert len(paths) == 1
    assert paths[0].suffix == ".jpg"
    assert paths[0].read_bytes().startswith(b"\xff\xd8")
    with Image.open(paths[0]) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.width >= 300
        assert image.height >= 150
