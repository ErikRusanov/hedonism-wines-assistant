"""Unit tests for the bottle-image import helper (SKU normalisation).

The script lives under ``data/`` (a loose data-track tool, not part of the
package), so it is loaded by path. Only the pure filename->SKU parsing is pinned
here; the copy/report side runs against the live filesystem.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "data" / "import_images.py"
_spec = importlib.util.spec_from_file_location("import_images", _MODULE_PATH)
import_images = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(import_images)


def test_plain_sku() -> None:
    assert import_images.sku_from_filename(Path("HED28846.JPG")) == "HED28846"


def test_strips_browser_duplicate_suffix() -> None:
    assert import_images.sku_from_filename(Path("HED11602 (1).JPG")) == "HED11602"
    assert import_images.sku_from_filename(Path("HED11602 (12).jpg")) == "HED11602"


def test_lowercase_extension_accepted() -> None:
    assert import_images.sku_from_filename(Path("HED7040.jpg")) == "HED7040"


def test_non_sku_filename_returns_none() -> None:
    assert import_images.sku_from_filename(Path("vacation-photo.jpg")) is None
    assert import_images.sku_from_filename(Path("IMG_1234.JPG")) is None
