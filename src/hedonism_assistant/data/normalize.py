"""Deterministic normalization: :class:`RawWine` -> canonical :class:`Wine` (I-2).

The scrape output (I-1) is intentionally permissive -- every field optional, raw
text kept verbatim. This module is the offline counterpart that turns those
records into the canonical product cards the rest of the system depends on:

* derive the catalogue ``category`` (still / sparkling / sweet / fortified),
  which the site does not expose as a field, from name/region/grape keywords;
* canonicalise grape synonyms (Tinto Fino -> Tempranillo) and a few country
  spellings, so payload filters and the taxonomy collapse onto one spelling;
* name the bottle ``format`` (1500ml -> Magnum) from its size;
* build the NL "passport" (:func:`build_embedding_text`) that dense retrieval
  embeds -- a coherent sentence or two assembled from the structured fields plus
  a bounded excerpt of the (copyrighted) tasting note.

Everything here is pure and deterministic: no network, no LLM. The optional LLM
pass (missing colour, style/food-pairing tags) lives in
:mod:`hedonism_assistant.data.enricher` and is layered on top by the orchestrator.
A record missing a field the canonical card requires (name, url, price, size) is
returned as ``None`` and counted as dropped rather than coerced into a bad card.
"""

from __future__ import annotations

import re
from typing import Final

from hedonism_assistant.data.models import RawWine
from hedonism_assistant.models.wine import Availability, Wine, WineCategory

# --------------------------------------------------------------------------- #
# Category classification                                                      #
# --------------------------------------------------------------------------- #
# The site has a colour facet but no still/sparkling/sweet/fortified one, so the
# category is inferred from identity keywords (name, region, grape, producer).
# Terms are matched on word boundaries against a casefolded haystack. Priority is
# fortified > sparkling > sweet > still: a vintage Port is sweet *and* fortified
# but belongs under fortified; a demi-sec Champagne is sweet *and* sparkling but
# belongs under sparkling.
_FORTIFIED_TERMS: Final[frozenset[str]] = frozenset(
    {
        "port",
        "porto",
        "tawny",
        "colheita",
        "lbv",
        "sherry",
        "jerez",
        "manzanilla",
        "amontillado",
        "oloroso",
        "palo cortado",
        "pedro ximénez",
        "pedro ximenez",
        "madeira",
        "sercial",
        "malmsey",
        "bual",
        "boal",
        "marsala",
        "vermouth",
        "banyuls",
        "maury",
        "rivesaltes",
        "rasteau",
        "vin doux naturel",
        "commandaria",
        "mavrodaphne",
    }
)
_SPARKLING_TERMS: Final[frozenset[str]] = frozenset(
    {
        "champagne",
        "crémant",
        "cremant",
        "prosecco",
        "cava",
        "franciacorta",
        "spumante",
        "espumante",
        "sparkling",
        "mousseux",
        "sekt",
        "blanquette",
        "lambrusco",
        "asti",
        "pétillant",
        "petillant",
        "pet-nat",
        "pét-nat",
        "metodo classico",
        "traditional method",
        "méthode champenoise",
        "trento",
    }
)
_SWEET_TERMS: Final[frozenset[str]] = frozenset(
    {
        "sauternes",
        "barsac",
        "tokaji",
        "tokay",
        "aszú",
        "aszu",
        "eiswein",
        "icewine",
        "ice wine",
        "beerenauslese",
        "trockenbeerenauslese",
        "vin santo",
        "vinsanto",
        "vin de paille",
        "passito",
        "recioto",
        "noble rot",
        "botrytis",
        "late harvest",
        "vendange tardive",
        "moelleux",
        "sélection de grains nobles",
        "grains nobles",
        "dessert wine",
        "sweet wine",
    }
)


def _compile_terms(terms: frozenset[str]) -> re.Pattern[str]:
    """One word-boundary alternation over all terms (multi-word terms included)."""
    alternation = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"\b(?:{alternation})\b")


_FORTIFIED_RE: Final = _compile_terms(_FORTIFIED_TERMS)
_SPARKLING_RE: Final = _compile_terms(_SPARKLING_TERMS)
_SWEET_RE: Final = _compile_terms(_SWEET_TERMS)


def _category_haystack(raw: RawWine) -> str:
    """Identity text used for category keywords: name + region only.

    Grapes and producer are deliberately excluded: they do not signal a category
    and they collide with style words (the grape "Tinto Fino" would otherwise hit
    the sherry term "fino"). Name and region carry the category in practice
    (Champagne, Sauternes, Port, Jerez).
    """
    return " ".join((raw.name or "", raw.region_raw or raw.region or "")).casefold()


def classify_category(raw: RawWine) -> WineCategory:
    """Infer the catalogue category from identity keywords (fortified-first)."""
    haystack = _category_haystack(raw)
    if _FORTIFIED_RE.search(haystack):
        return WineCategory.FORTIFIED
    if _SPARKLING_RE.search(haystack):
        return WineCategory.SPARKLING
    if _SWEET_RE.search(haystack):
        return WineCategory.SWEET
    return WineCategory.STILL


# --------------------------------------------------------------------------- #
# Grape & country canonicalisation                                             #
# --------------------------------------------------------------------------- #
# Casefolded synonym -> canonical spelling. Folding regional/translated synonyms
# onto one name lifts filter recall: a search for "Syrah" should also match a
# bottle the catalogue labels "Shiraz".
_GRAPE_SYNONYMS: Final[dict[str, str]] = {
    "tinto fino": "Tempranillo",
    "tinta del pais": "Tempranillo",
    "tinta del país": "Tempranillo",
    "tinta de toro": "Tempranillo",
    "tinta roriz": "Tempranillo",
    "aragonez": "Tempranillo",
    "aragonês": "Tempranillo",
    "ull de llebre": "Tempranillo",
    "cencibel": "Tempranillo",
    "pinot nero": "Pinot Noir",
    "spätburgunder": "Pinot Noir",
    "spatburgunder": "Pinot Noir",
    "blauburgunder": "Pinot Noir",
    "pinot grigio": "Pinot Gris",
    "grauburgunder": "Pinot Gris",
    "ruländer": "Pinot Gris",
    "weissburgunder": "Pinot Blanc",
    "weißburgunder": "Pinot Blanc",
    "pinot bianco": "Pinot Blanc",
    "shiraz": "Syrah",
    "garnacha": "Grenache",
    "grenache noir": "Grenache",
    "cannonau": "Grenache",
    "garnacha blanca": "Grenache Blanc",
    "garnatxa blanca": "Grenache Blanc",
    "monastrell": "Mourvèdre",
    "mataro": "Mourvèdre",
    "mourvedre": "Mourvèdre",
    "mazuelo": "Carignan",
    "cariñena": "Carignan",
    "carinena": "Carignan",
    "carignane": "Carignan",
    "brunello": "Sangiovese",
    "prugnolo gentile": "Sangiovese",
    "morellino": "Sangiovese",
    "sangioveto": "Sangiovese",
    "nielluccio": "Sangiovese",
    "spanna": "Nebbiolo",
    "chiavennasca": "Nebbiolo",
    "ugni blanc": "Trebbiano",
    "alvarinho": "Albariño",
    "moscatel": "Muscat",
    "moscato": "Muscat",
    "primitivo": "Zinfandel",
    "côt": "Malbec",
    "cot": "Malbec",
    "gewurztraminer": "Gewürztraminer",
    "gewürztraminer": "Gewürztraminer",
}

_COUNTRY_SYNONYMS: Final[dict[str, str]] = {
    "usa": "USA",
    "u.s.a.": "USA",
    "us": "USA",
    "u.s.": "USA",
    "united states": "USA",
    "united states of america": "USA",
    "america": "USA",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "great britain": "United Kingdom",
}

_WS_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


def _clean(text: str | None) -> str | None:
    """Trim and collapse whitespace; empty -> None."""
    if not text:
        return None
    collapsed = _WS_RE.sub(" ", text).strip()
    return collapsed or None


def canonicalize_grapes(grapes: list[str]) -> list[str]:
    """Map known synonyms onto canonical names; dedupe, preserve order."""
    canonical = (
        _GRAPE_SYNONYMS.get(cleaned.casefold(), cleaned)
        for grape in grapes
        if (cleaned := _clean(grape)) is not None
    )
    # dict.fromkeys keeps first-seen order while collapsing duplicates.
    return list(dict.fromkeys(canonical))


def canonicalize_country(country: str | None) -> str | None:
    """Collapse a few country spellings (USA variants, UK) onto one form."""
    cleaned = _clean(country)
    if cleaned is None:
        return None
    return _COUNTRY_SYNONYMS.get(cleaned.casefold(), cleaned)


# --------------------------------------------------------------------------- #
# Bottle format                                                                #
# --------------------------------------------------------------------------- #
# Named large/small formats by volume. 750ml is the standard bottle and gets no
# name (left as None); sizes outside the table keep their millilitre value only.
_BOTTLE_FORMATS: Final[dict[int, str]] = {
    187: "Piccolo",
    375: "Half",
    1500: "Magnum",
    3000: "Double Magnum",
    4500: "Jeroboam",
    6000: "Imperial",
    9000: "Salmanazar",
    12000: "Balthazar",
    15000: "Nebuchadnezzar",
}


def format_name(bottle_size_ml: int) -> str | None:
    """Named bottle format for a volume, or ``None`` for a standard 750ml bottle."""
    return _BOTTLE_FORMATS.get(bottle_size_ml)


# --------------------------------------------------------------------------- #
# Normalization entry point                                                    #
# --------------------------------------------------------------------------- #
def normalize_wine(raw: RawWine) -> Wine | None:
    """Turn one scrape record into a canonical :class:`Wine`.

    Returns ``None`` (a drop) when the record is not a wine or lacks a field the
    canonical card requires (name, url, price, bottle size). ``embedding_text`` is
    deliberately left unset here -- the orchestrator fills it with
    :func:`build_embedding_text` after any LLM enrichment, so style/pairing tags
    can make it into the passport.
    """
    if not raw.is_wine:
        return None

    identifier = raw.sku or raw.slug
    name = _clean(raw.name)
    if not identifier or not name or raw.price is None or raw.bottle_size_ml is None:
        return None

    return Wine(
        id=identifier,
        slug=raw.slug,
        name=name,
        url=raw.url,
        category=classify_category(raw),
        color=raw.color,
        producer=_clean(raw.producer),
        country=canonicalize_country(raw.country),
        region=_clean(raw.region),
        sub_region=_clean(raw.sub_region),
        vintage=raw.vintage,
        grapes=canonicalize_grapes(raw.grapes),
        abv=raw.abv,
        bottle_size_ml=raw.bottle_size_ml,
        format_name=format_name(raw.bottle_size_ml),
        price=raw.price,
        price_ex_vat=raw.price_ex_vat,
        currency=raw.currency,
        on_sale=raw.on_sale,
        sale_was_price=raw.sale_was_price,
        in_bond=raw.in_bond,
        is_vegan=raw.is_vegan,
        is_organic=raw.is_organic,
        is_kosher=raw.is_kosher,
        is_alcohol_free=raw.is_alcohol_free,
        availability=raw.availability or Availability.IN_STOCK,
        stock_qty=raw.stock_qty,
        critic_scores=raw.critic_scores,
        tasting_notes=raw.tasting_notes,
        image_url=raw.image_url,
    )


# --------------------------------------------------------------------------- #
# Embedding passport                                                           #
# --------------------------------------------------------------------------- #
_CURRENCY_SYMBOLS: Final[dict[str, str]] = {"GBP": "£", "EUR": "€", "USD": "$"}
_SENTENCE_END_RE: Final[re.Pattern[str]] = re.compile(r"[.!?]")


def _format_price(price: float, currency: str) -> str:
    symbol = _CURRENCY_SYMBOLS.get(currency.upper())
    amount = f"{price:,.0f}" if float(price).is_integer() else f"{price:,.2f}"
    return f"{symbol}{amount}" if symbol else f"{amount} {currency}"


def _join_natural(items: list[str]) -> str:
    """'A', 'A and B', 'A, B and C'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _truncate_notes(notes: str, limit: int) -> str:
    """Trim tasting notes to ~``limit`` chars, preferring a sentence boundary."""
    collapsed = _WS_RE.sub(" ", notes).strip()
    if len(collapsed) <= limit:
        return collapsed
    window = collapsed[:limit]
    boundary = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if boundary >= limit // 2:
        return window[: boundary + 1]
    return window.rsplit(" ", 1)[0].rstrip(",;:") + "…"


def build_embedding_text(wine: Wine, *, notes_chars: int = 600) -> str:
    """Assemble the NL passport embedded for dense retrieval.

    A few short, factual sentences from the structured fields, followed by a
    bounded excerpt of the tasting note. Deterministic and self-contained so the
    same card always yields the same vector input.
    """
    # Identity sentence.
    descriptor = " ".join(
        part
        for part in (
            str(wine.vintage) if wine.vintage else "",
            wine.color.value if wine.color else "",
            wine.category.value,
        )
        if part
    )
    identity = f"{wine.name} is a {descriptor} wine"
    origin = ", ".join(p for p in (wine.sub_region, wine.region, wine.country) if p)
    if origin:
        identity += f" from {origin}"
    if wine.producer:
        identity += f", produced by {wine.producer}"
    if wine.grapes:
        lead = "a blend of " if len(wine.grapes) > 1 else ""
        identity += f", made from {lead}{_join_natural(wine.grapes)}"
    sentences = [identity + "."]

    # Specs sentence.
    specs: list[str] = []
    if wine.abv is not None:
        specs.append(f"{wine.abv:g}% ABV")
    size = (
        f"a {wine.bottle_size_ml} ml {wine.format_name} bottle"
        if wine.format_name
        else f"a {wine.bottle_size_ml} ml bottle"
    )
    specs.append(size)
    specs.append(f"priced at {_format_price(wine.price, wine.currency)}")
    if wine.in_bond:
        specs.append("sold in bond")
    sentences.append("It is " + _join_natural(specs) + ".")

    # Dietary / production flags sentence (only the badges the product carries).
    diet = [
        phrase
        for flag, phrase in (
            (wine.is_vegan, "suitable for vegans"),
            (wine.is_organic, "organic"),
            (wine.is_kosher, "kosher"),
            (wine.is_alcohol_free, "alcohol-free"),
        )
        if flag
    ]
    if diet:
        sentences.append(f"It is {_join_natural(diet)}.")

    # Critic sentence.
    if wine.critic_scores:
        rated = _join_natural([f"{s.score:g}/{s.scale} ({s.critic})" for s in wine.critic_scores])
        sentences.append(f"Critics rate it {rated}.")

    # Enrichment tags (present only after LLM enrichment).
    if wine.style_tags:
        sentences.append(f"Style: {_join_natural(wine.style_tags)}.")
    if wine.food_pairings:
        sentences.append(f"Pairs with {_join_natural(wine.food_pairings)}.")

    # Tasting note excerpt.
    if wine.tasting_notes and notes_chars > 0:
        excerpt = _truncate_notes(wine.tasting_notes, notes_chars)
        if excerpt:
            sentences.append(excerpt)

    return " ".join(sentences)
