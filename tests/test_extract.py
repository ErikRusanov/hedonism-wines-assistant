"""Tests for the HTML extraction orchestrator over a temp cache dir."""

import shutil
from pathlib import Path

from hedonism_assistant.config import Settings
from hedonism_assistant.data.extract import run_extract
from hedonism_assistant.models.wine import Wine

FIXTURES = Path(__file__).parent / "fixtures"


def _seed(html_dir: Path, names: dict[str, str]) -> None:
    """Copy committed fixture HTML into ``html_dir`` under given <slug>.html names."""
    html_dir.mkdir(parents=True, exist_ok=True)
    for dest_slug, fixture in names.items():
        shutil.copy(FIXTURES / fixture, html_dir / f"{dest_slug}.html")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        html_input_dir=str(tmp_path / "html"),
        extract_output_path=str(tmp_path / "wines.enriched.jsonl"),
    )


def test_extract_writes_cards_and_drops_spirit(tmp_path: Path) -> None:
    _seed(
        tmp_path / "html",
        {"a-wine": "product_wine.html", "a-spirit": "product_spirit.html"},
    )
    settings = _settings(tmp_path)

    report = run_extract(settings)

    assert report.read == 2
    assert report.non_wine_skipped == 1
    assert report.written == 1
    out = Path(settings.extract_output_path)
    assert out.exists()
    cards = [Wine.model_validate_json(line) for line in out.read_text().splitlines()]
    assert len(cards) == 1
    wine = cards[0]
    # The passport is attached during extraction (the indexer relies on it).
    assert wine.embedding_text
    assert wine.name in wine.embedding_text


def test_extract_dedupes_same_product(tmp_path: Path) -> None:
    # Two files, same underlying product (same SKU) -> one card, one duplicate.
    _seed(
        tmp_path / "html",
        {"wine-a": "product_wine.html", "wine-a-copy": "product_wine.html"},
    )
    settings = _settings(tmp_path)

    report = run_extract(settings)

    assert report.read == 2
    assert report.duplicates == 1
    assert report.written == 1


def test_extract_report_has_coverage_and_categories(tmp_path: Path) -> None:
    _seed(tmp_path / "html", {"a-wine": "product_wine.html"})

    report = run_extract(_settings(tmp_path))

    assert report.category_counts  # at least one category counted
    assert set(report.field_coverage) == set(report.COVERAGE_FIELDS)
    assert report.field_coverage["embedding_text"] == 1.0


def test_extract_limit_caps_files(tmp_path: Path) -> None:
    _seed(
        tmp_path / "html",
        {"a-wine": "product_wine.html", "b-spirit": "product_spirit.html"},
    )

    report = run_extract(_settings(tmp_path), limit=1)

    assert report.read == 1
