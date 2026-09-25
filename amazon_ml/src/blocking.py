from __future__ import annotations

import re
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple

import pandas as pd

from .normalize import (
    normalize_basic,
    normalize_compact,
    remove_legal_suffix,
    transliterate,
    extract_numbers,
)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_EXACT_BLOCK = 500
MAX_PREFIX_BLOCK = 250
MAX_NAME_TOKEN_BLOCK = 150
MAX_TRANSLIT_TOKEN_BLOCK = 150
MAX_ADDRESS_TOKEN_BLOCK = 150
MAX_ADDRESS_NGRAM_BLOCK = 0
MAX_ADDRESS_PAIR_BLOCK = 0


# ============================================================
# GENERIC TOKENS
# ============================================================

GENERIC_NAME_TOKENS = {
    "the",
    "and",
    "company",
    "corporation",
    "corporate",
    "limited",
    "private",
    "pvt",
    "ltd",
    "llc",
    "inc",
    "incorporated",
    "co",
    "group",
    "services",
    "service",
    "solutions",
    "solution",
    "enterprises",
    "enterprise",
    "business",
    "businesses",
    "international",
    "india",
    "america",
    "american",
    "usa",
    "united",
    "global",
    "official",
    "shop",
    "store",
}

GENERIC_ADDRESS_TOKENS = {
    "road",
    "street",
    "st",
    "rd",
    "avenue",
    "ave",
    "lane",
    "ln",
    "drive",
    "dr",
    "highway",
    "hwy",
    "circle",
    "place",
    "pl",
    "building",
    "block",
    "floor",
    "near",
    "opposite",
    "opp",
    "main",
    "india",
    "usa",
    "america",
}

WEAK_PREFIX_TOKENS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "sri",
    "shri",
    "smt",
    "the",
    "m",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_string(value) -> str:
    """Convert a possibly missing value into a clean string."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def normalize_country(country: str) -> str:
    return safe_string(country).lower().strip()


# ============================================================
# NAME HELPERS
# ============================================================

def get_name_tokens(name: str) -> List[str]:
    """
    Informative tokens from the basic normalized name.
    """
    name = normalize_basic(name)

    if not name:
        return []

    tokens = name.split()

    result = []

    for token in tokens:

        if len(token) < 3:
            continue

        if token in GENERIC_NAME_TOKENS:
            continue

        result.append(token)

    return result


def get_translit_tokens(name: str) -> List[str]:
    """
    Informative Latin transliterated tokens.

    Useful when the same business appears in different scripts.
    """
    name = transliterate(name)

    if not name:
        return []

    tokens = name.split()

    result = []

    for token in tokens:

        if len(token) < 3:
            continue

        if token in GENERIC_NAME_TOKENS:
            continue

        result.append(token)

    return result


def get_name_prefix(
    name: str,
    length: int = 5,
) -> str:

    compact = normalize_compact(name)

    if not compact:
        return ""

    return compact[:length]


def get_translit_prefix(
    name: str,
    length: int = 6,
) -> str:

    translit_name = transliterate(name)

    if not translit_name:
        return ""

    compact = re.sub(
        r"[^a-z0-9]",
        "",
        translit_name,
    )

    return compact[:length]


# ============================================================
# ADDRESS HELPERS
# ============================================================

def get_address_tokens(address: str) -> List[str]:
    """
    Informative address tokens.
    """
    address = normalize_basic(address)

    if not address:
        return []

    tokens = address.split()

    result = []

    for token in tokens:

        if len(token) < 4:
            continue

        if token in GENERIC_ADDRESS_TOKENS:
            continue

        result.append(token)

    return result


def get_address_token_pairs(
    address: str,
) -> List[Tuple[str, str]]:
    """
    Build combinations of informative address tokens.

    Example:

        '911 fifth street chickasha oklahoma'

    can produce:

        ('fifth', 'chickasha')
        ('chickasha', 'oklahoma')
        ...
    """
    tokens = get_address_tokens(address)

    pairs = set()

    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):

            a = tokens[i]
            b = tokens[j]

            if a == b:
                continue

            pairs.add(
                tuple(sorted((a, b)))
            )

    return list(pairs)


def get_address_number_key(
    address: str,
) -> str:

    numbers = extract_numbers(address)

    if not numbers:
        return ""

    for number in numbers:

        if 1 <= len(number) <= 5:
            return number

    return numbers[0]


def get_postal_codes(
    address: str,
) -> List[str]:

    return [
        number
        for number in extract_numbers(address)
        if len(number) in (5, 6)
    ]


def get_char_ngrams(
    token: str,
    n: int = 3,
) -> Set[str]:
    """
    Character n-grams for typo-tolerant retrieval.

    Example:

        chickasha

    produces:

        chi, hic, ick, cka, kas, ash, sha
    """
    token = safe_string(token)

    if len(token) < n:
        return set()

    return {
        token[i:i + n]
        for i in range(len(token) - n + 1)
    }


# ============================================================
# INDEX
# ============================================================

class BlockingIndex:

    def __init__(self):

        # ----------------------------
        # Names
        # ----------------------------

        self.exact_name: Dict[str, List[str]] = defaultdict(list)

        self.name_no_suffix: Dict[str, List[str]] = defaultdict(list)

        self.name_translit: Dict[str, List[str]] = defaultdict(list)

        self.name_translit_no_suffix: Dict[str, List[str]] = defaultdict(list)

        self.name_prefix5: Dict[str, List[str]] = defaultdict(list)

        self.name_prefix8: Dict[str, List[str]] = defaultdict(list)

        self.name_token: Dict[str, List[str]] = defaultdict(list)

        self.name_translit_token: Dict[str, List[str]] = defaultdict(list)

        self.name_translit_prefix: Dict[str, List[str]] = defaultdict(list)

        # ----------------------------
        # Address
        # ----------------------------

        self.address_number: Dict[str, List[str]] = defaultdict(list)

        self.postal_code: Dict[str, List[str]] = defaultdict(list)

        self.address_token: Dict[str, List[str]] = defaultdict(list)

        self.address_token_pair: Dict[str, List[str]] = defaultdict(list)

        self.address_ngram: Dict[str, List[str]] = defaultdict(list)

        self.records = 0

    # ========================================================
    # ADD RECORD
    # ========================================================

    def add_record(
        self,
        entity_id: str,
        business_name: str,
        business_address: str,
        country: str,
    ) -> None:

        entity_id = safe_string(entity_id)
        business_name = safe_string(business_name)
        business_address = safe_string(business_address)
        country = normalize_country(country)

        self.records += 1

        # ====================================================
        # NAME
        # ====================================================

        name_basic = normalize_basic(
            business_name
        )

        name_no_suffix = remove_legal_suffix(
            business_name
        )

        name_translit = transliterate(
            business_name
        )

        name_translit_no_suffix = transliterate(
            name_no_suffix
        )

        # ----------------------------------------------------
        # Exact basic name
        # ----------------------------------------------------

        if name_basic:

            self.exact_name[
                f"{country}|{name_basic}"
            ].append(entity_id)

        # ----------------------------------------------------
        # Suffix stripped
        # ----------------------------------------------------

        if name_no_suffix:

            self.name_no_suffix[
                f"{country}|{name_no_suffix}"
            ].append(entity_id)

        # ----------------------------------------------------
        # Exact transliteration
        # ----------------------------------------------------

        if name_translit:

            self.name_translit[
                f"{country}|{name_translit}"
            ].append(entity_id)

        # ----------------------------------------------------
        # Transliteration without suffix
        # ----------------------------------------------------

        if name_translit_no_suffix:

            self.name_translit_no_suffix[
                f"{country}|{name_translit_no_suffix}"
            ].append(entity_id)

        # ----------------------------------------------------
        # Basic prefixes
        # ----------------------------------------------------

        prefix5 = get_name_prefix(
            name_basic,
            5,
        )

        if prefix5:

            self.name_prefix5[
                f"{country}|{prefix5}"
            ].append(entity_id)

        prefix8 = get_name_prefix(
            name_basic,
            8,
        )

        if prefix8:

            self.name_prefix8[
                f"{country}|{prefix8}"
            ].append(entity_id)

        # ----------------------------------------------------
        # Basic informative tokens
        # ----------------------------------------------------

        for token in get_name_tokens(
            business_name
        ):

            self.name_token[
                f"{country}|{token}"
            ].append(entity_id)

        # ----------------------------------------------------
        # Transliteration tokens
        # ----------------------------------------------------

        for token in get_translit_tokens(
            business_name
        ):

            self.name_translit_token[
                f"{country}|{token}"
            ].append(entity_id)

        # ----------------------------------------------------
        # Transliteration compact prefix
        # ----------------------------------------------------

        translit_prefix = get_translit_prefix(
            business_name,
            6,
        )

        if translit_prefix:

            self.name_translit_prefix[
                f"{country}|{translit_prefix}"
            ].append(entity_id)

        # ====================================================
        # ADDRESS
        # ====================================================

        number = get_address_number_key(
            business_address
        )

        # ----------------------------------------------------
        # House/building number
        # ----------------------------------------------------

        if number:

            self.address_number[
                f"{country}|{number}"
            ].append(entity_id)

        # ----------------------------------------------------
        # Postal code
        # ----------------------------------------------------

        for postal in get_postal_codes(
            business_address
        ):

            self.postal_code[
                f"{country}|{postal}"
            ].append(entity_id)

        # ----------------------------------------------------
        # Address tokens
        # ----------------------------------------------------

        address_tokens = get_address_tokens(
            business_address
        )

        for token in address_tokens:

            self.address_token[
                f"{country}|{token}"
            ].append(entity_id)

            # Character n-grams
            if len(token) >= 5:

                for gram in get_char_ngrams(
                    token,
                    3,
                ):

                    self.address_ngram[
                        f"{country}|{gram}"
                    ].append(entity_id)

        # ----------------------------------------------------
        # Address token pairs
        # ----------------------------------------------------

        for a, b in get_address_token_pairs(
            business_address
        ):

            self.address_token_pair[
                f"{country}|{a}|{b}"
            ].append(entity_id)


# ============================================================
# CANDIDATE HELPER
# ============================================================

def _add_candidates(
    destination: Set[str],
    index: Dict[str, List[str]],
    key: str,
    max_size: int,
) -> None:

    if not key:
        return

    values = index.get(key)

    if not values:
        return

    # Ignore blocks that are too broad.
    if len(values) > max_size:
        return

    destination.update(values)

def _add_overlap_candidates(
    destination: Set[str],
    index: Dict[str, List[str]],
    keys: List[str],
    minimum_overlap: int,
    max_posting_size: int,
) -> None:
    """
    Add candidates only when a record shares multiple blocking
    signals with the query.

    This is deliberately stricter than normal blocking.

    Example:
        address tokens:
        ['new', 'delhi']

    A candidate must share both signals to be added.
    """

    overlap_count = Counter()

    for key in keys:

        if not key:
            continue

        values = index.get(key)

        if not values:
            continue

        # Ignore extremely common values.
        if len(values) > max_posting_size:
            continue

        for entity_id in values:
            overlap_count[entity_id] += 1

    for entity_id, count in overlap_count.items():

        if count >= minimum_overlap:
            destination.add(entity_id)

# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates_for_record(
    s1_row,
    index: BlockingIndex,
) -> Set[str]:

    candidates: Set[str] = set()

    entity_id = safe_string(
        s1_row.get("entity_id")
    )

    business_name = safe_string(
        s1_row.get("business_name")
    )

    business_address = safe_string(
        s1_row.get("business_address")
    )

    country = normalize_country(
        s1_row.get("country")
    )

    # ========================================================
    # NAME
    # ========================================================

    name_basic = normalize_basic(
        business_name
    )

    name_no_suffix = remove_legal_suffix(
        business_name
    )

    name_translit = transliterate(
        business_name
    )

    name_translit_no_suffix = transliterate(
        name_no_suffix
    )

    # --------------------------------------------------------
    # Exact name
    # --------------------------------------------------------

    if name_basic:

        _add_candidates(
            candidates,
            index.exact_name,
            f"{country}|{name_basic}",
            MAX_EXACT_BLOCK,
        )

    # --------------------------------------------------------
    # Suffix-stripped name
    # --------------------------------------------------------

    if name_no_suffix:

        _add_candidates(
            candidates,
            index.name_no_suffix,
            f"{country}|{name_no_suffix}",
            MAX_EXACT_BLOCK,
        )

    # --------------------------------------------------------
    # Exact transliteration
    # --------------------------------------------------------

    if name_translit:

        _add_candidates(
            candidates,
            index.name_translit,
            f"{country}|{name_translit}",
            MAX_EXACT_BLOCK,
        )

    # --------------------------------------------------------
    # Transliteration without suffix
    # --------------------------------------------------------

    if name_translit_no_suffix:

        _add_candidates(
            candidates,
            index.name_translit_no_suffix,
            f"{country}|{name_translit_no_suffix}",
            MAX_EXACT_BLOCK,
        )

    # --------------------------------------------------------
    # Normalized name prefix
    # --------------------------------------------------------

    prefix5 = get_name_prefix(
        name_basic,
        5,
    )

    if prefix5:

        _add_candidates(
            candidates,
            index.name_prefix5,
            f"{country}|{prefix5}",
            MAX_PREFIX_BLOCK,
        )

    prefix8 = get_name_prefix(
        name_basic,
        8,
    )

    if prefix8:

        _add_candidates(
            candidates,
            index.name_prefix8,
            f"{country}|{prefix8}",
            MAX_PREFIX_BLOCK,
        )

    # --------------------------------------------------------
    # Name tokens
    # --------------------------------------------------------

    for token in get_name_tokens(
        business_name
    ):

        _add_candidates(
            candidates,
            index.name_token,
            f"{country}|{token}",
            MAX_NAME_TOKEN_BLOCK,
        )

    # --------------------------------------------------------
    # Transliteration tokens
    # --------------------------------------------------------

    for token in get_translit_tokens(
        business_name
    ):

        _add_candidates(
            candidates,
            index.name_translit_token,
            f"{country}|{token}",
            MAX_TRANSLIT_TOKEN_BLOCK,
        )

    # --------------------------------------------------------
    # Transliteration prefix
    # --------------------------------------------------------

    translit_prefix = get_translit_prefix(
        business_name,
        6,
    )

    if translit_prefix:

        _add_candidates(
            candidates,
            index.name_translit_prefix,
            f"{country}|{translit_prefix}",
            MAX_PREFIX_BLOCK,
        )

    # ========================================================
    # ADDRESS
    # ========================================================
    address_overlap_keys = []
    address_ngram_keys = []
    
    number = get_address_number_key(
        business_address
    )

    # --------------------------------------------------------
    # Address number
    # --------------------------------------------------------

    if number:

        _add_candidates(
            candidates,
            index.address_number,
            f"{country}|{number}",
            MAX_EXACT_BLOCK,
        )

    # --------------------------------------------------------
    # Postal code
    # --------------------------------------------------------

    for postal in get_postal_codes(
        business_address
    ):

        _add_candidates(
            candidates,
            index.postal_code,
            f"{country}|{postal}",
            MAX_EXACT_BLOCK,
        )

    # --------------------------------------------------------
    # Address tokens + character ngrams
    # --------------------------------------------------------

    address_tokens = get_address_tokens(
        business_address
    )

    for token in address_tokens:

        key = f"{country}|{token}"

        _add_candidates(
            candidates,
            index.address_token,
            key,
            MAX_ADDRESS_TOKEN_BLOCK,
        )

        address_overlap_keys.append(key)

        if len(token) >= 5:

            for gram in get_char_ngrams(
                token,
                3,
            ):

                address_ngram_keys.append(
                    f"{country}|{gram}"
                )


    _add_overlap_candidates(
        candidates,
        index.address_token,
        address_overlap_keys,
        minimum_overlap=2,
        max_posting_size=1000,
    )

    _add_overlap_candidates(
        candidates,
        index.address_ngram,
        address_ngram_keys,
        minimum_overlap=2,
        max_posting_size=300,
    )


    # --------------------------------------------------------
    # Address token pairs
    # --------------------------------------------------------

    for a, b in get_address_token_pairs(
        business_address
    ):

        _add_candidates(
            candidates,
            index.address_token_pair,
            f"{country}|{a}|{b}",
            MAX_ADDRESS_PAIR_BLOCK,
        )

    # --------------------------------------------------------
    # Never return self
    # --------------------------------------------------------

    candidates.discard(entity_id)

    return candidates


# ============================================================
# DATAFRAME UTILITIES
# ============================================================

def build_index(
    df: pd.DataFrame,
) -> BlockingIndex:

    required_columns = {
        "entity_id",
        "business_name",
        "business_address",
        "country",
    }

    missing = required_columns - set(
        df.columns
    )

    if missing:

        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    index = BlockingIndex()

    for row in df.itertuples(
        index=False
    ):

        index.add_record(
            entity_id=getattr(
                row,
                "entity_id",
            ),
            business_name=getattr(
                row,
                "business_name",
            ),
            business_address=getattr(
                row,
                "business_address",
            ),
            country=getattr(
                row,
                "country",
            ),
        )

    return index


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    index_s2: BlockingIndex,
    index_s3: BlockingIndex,
) -> pd.DataFrame:

    rows: List[
        Tuple[str, str]
    ] = []

    for row in s1_df.itertuples(
        index=False
    ):

        s1_id = safe_string(
            getattr(
                row,
                "entity_id",
            )
        )

        s1_record = {
            "entity_id": s1_id,
            "business_name": getattr(
                row,
                "business_name",
            ),
            "business_address": getattr(
                row,
                "business_address",
            ),
            "country": getattr(
                row,
                "country",
            ),
        }

        s2_candidates = (
            generate_candidates_for_record(
                s1_record,
                index_s2,
            )
        )

        s3_candidates = (
            generate_candidates_for_record(
                s1_record,
                index_s3,
            )
        )

        for candidate in s2_candidates:

            rows.append(
                (
                    s1_id,
                    candidate,
                )
            )

        for candidate in s3_candidates:

            rows.append(
                (
                    s1_id,
                    candidate,
                )
            )

    return pd.DataFrame(
        rows,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
        ],
    )


# ============================================================
# SMOKE TEST
# ============================================================

if __name__ == "__main__":

    print("Testing blocking.py V3 ...")

    s2 = pd.DataFrame([
        {
            "entity_id": "S2-001",
            "business_name":
                "Shree Ganesh Electronics Pvt Ltd",
            "business_address":
                "12 MG Road Bengaluru 560037",
            "country":
                "India",
        },
        {
            "entity_id": "S2-002",
            "business_name":
                "ABC Electronics",
            "business_address":
                "99 Main Road Delhi 110001",
            "country":
                "India",
        },
        {
            "entity_id": "S2-003",
            "business_name":
                "Sri Shree Ganesh Electronics",
            "business_address":
                "12 MG Rd Bengaluru 560037",
            "country":
                "India",
        },
        {
            "entity_id": "S2-004",
            "business_name":
                "श्री गणेश इलेक्ट्रॉनिक्स",
            "business_address":
                "12 MG Road Bengaluru 560037",
            "country":
                "India",
        },
    ])

    s3 = pd.DataFrame([
        {
            "entity_id": "S3-001",
            "business_name":
                "Shree Ganesh Electronics",
            "business_address":
                "12 M G Rd Bengaluru 560037",
            "country":
                "India",
        },
        {
            "entity_id": "S3-002",
            "business_name":
                "XYZ Corporation",
            "business_address":
                "50 MG Road Bengaluru 560002",
            "country":
                "India",
        },
        {
            "entity_id": "S3-003",
            "business_name":
                "Ganesh Electronics",
            "business_address":
                "12 MG Road Bengaluru 560037",
            "country":
                "India",
        },
    ])

    s1 = pd.DataFrame([
        {
            "entity_id":
                "S1-001",
            "business_name":
                "Shree Ganesh Electronics Pvt. Ltd.",
            "business_address":
                "12, M.G. Road, Bengaluru 560037",
            "country":
                "India",
        }
    ])

    print("\nBuilding indexes...")

    index_s2 = build_index(s2)
    index_s3 = build_index(s3)

    print("\nGenerating candidates...")

    candidates = generate_candidate_pairs(
        s1,
        index_s2,
        index_s3,
    )

    print("\nCandidates:")

    print(
        candidates.to_string(
            index=False
        )
    )