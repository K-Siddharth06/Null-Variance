from __future__ import annotations

import re
import time
from collections import defaultdict

import pandas as pd

from .normalize import (
    normalize_basic,
    remove_legal_suffix,
    extract_numbers,
)


# ============================================================
# DATASET PATHS
# ============================================================

BASE = (
    r"C:\Users\darsh\Downloads"
    r"\6ab10eb3b23ba_student_resource"
    r"\student_resource\dataset\train"
)

S2_FILE = BASE + r"\train_source2.tsv"
S3_FILE = BASE + r"\train_source3.tsv"

S1_SAMPLE = r"diagnostics\sample_s1.tsv"
GT_SAMPLE = r"diagnostics\sample_ground_truth.tsv"

CHUNK_SIZE = 250_000


# ============================================================
# BLOCK LIMITS
# ============================================================

LIMITS = {
    "exact_name": 500,
    "name_no_suffix": 500,
    "prefix5": 50,
    "name_token": 50,
    "address_number": 50,
    "postal": 100,
    "address_token": 50,
}


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
    "mr",
    "mrs",
    "ms",
    "dr",
    "sri",
    "shri",
    "smt",
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
    "building",
    "block",
    "floor",
    "near",
    "opposite",
    "opp",
    "main",
}


# ============================================================
# NORMALIZATION
# ============================================================

def safe(value) -> str:

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def normalize_country(value) -> str:

    return safe(value).lower().strip()


def normalize_name(value) -> str:

    return normalize_basic(
        safe(value)
    )


def name_no_suffix(value) -> str:

    return remove_legal_suffix(
        safe(value)
    )


def compact_name(value) -> str:

    return re.sub(
        r"\s+",
        "",
        normalize_name(value),
    )


def name_tokens(value) -> list[str]:

    text = normalize_name(value)

    if not text:
        return []

    return [
        token
        for token in text.split()
        if len(token) >= 3
        and token not in GENERIC_NAME_TOKENS
    ]


def address_tokens(value) -> list[str]:

    text = normalize_basic(
        safe(value)
    )

    if not text:
        return []

    return [
        token
        for token in text.split()
        if len(token) >= 4
        and token not in GENERIC_ADDRESS_TOKENS
    ]


def first_number(value) -> str:

    numbers = extract_numbers(
        safe(value)
    )

    if not numbers:
        return ""

    for number in numbers:

        if len(number) <= 5:
            return number

    return numbers[0]


def postal_codes(value) -> list[str]:

    return [
        number
        for number in extract_numbers(
            safe(value)
        )
        if len(number) in (5, 6)
    ]


# ============================================================
# QUERY KEY GENERATION
# ============================================================

def generate_keys(row) -> dict[str, list[str]]:

    country = normalize_country(
        row["country"]
    )

    name = safe(
        row["business_name"]
    )

    address = safe(
        row["business_address"]
    )

    basic = normalize_name(name)

    no_suffix = name_no_suffix(name)

    compact = compact_name(name)

    keys = defaultdict(list)

    # Exact name
    if basic:

        keys["exact_name"].append(
            f"{country}|{basic}"
        )

    # Suffix stripped
    if no_suffix:

        keys["name_no_suffix"].append(
            f"{country}|{no_suffix}"
        )

    # Prefix
    if compact:

        keys["prefix5"].append(
            f"{country}|{compact[:5]}"
        )

    # Informative name tokens
    for token in name_tokens(name):

        keys["name_token"].append(
            f"{country}|{token}"
        )

    # Address number
    number = first_number(address)

    if number:

        keys["address_number"].append(
            f"{country}|{number}"
        )

    # Postal
    for postal in postal_codes(address):

        keys["postal"].append(
            f"{country}|{postal}"
        )

    # Address tokens
    for token in address_tokens(address):

        keys["address_token"].append(
            f"{country}|{token}"
        )

    return keys


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

def load_validation():

    print("Loading validation data...")

    s1 = pd.read_csv(
        S1_SAMPLE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    gt = pd.read_csv(
        GT_SAMPLE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = {}

    for row in gt.itertuples(
        index=False
    ):

        matched = safe(
            row.matched_entity_ids
        )

        if not matched:

            truth[
                row.source1_entity_id
            ] = set()

        else:

            truth[
                row.source1_entity_id
            ] = {
                x.strip()
                for x in matched.split(",")
                if x.strip()
            }

    return s1, truth


# ============================================================
# CAPPED INDEX
# ============================================================

class CappedIndex:

    def __init__(self, limit: int):

        self.limit = limit

        # key -> list of entity IDs
        #
        # None means the block became too large.
        self.data = {}

    def add(
        self,
        key: str,
        entity_id: str,
    ):

        current = self.data.get(
            key
        )

        if current is None:

            # Either not seen yet, or already overflowed.
            if key in self.data:
                return

            self.data[key] = [
                entity_id
            ]

            return

        # Already reached capacity.
        if len(current) >= self.limit:

            # Mark as unusable.
            self.data[key] = None

            return

        current.append(
            entity_id
        )

    def get(
        self,
        key: str,
    ):

        values = self.data.get(
            key
        )

        if values is None:
            return []

        return values


# ============================================================
# BUILD SELECTIVE INDEX
# ============================================================

def build_source_index(
    filename: str,
    source_name: str,
):

    print()
    print("=" * 70)
    print(f"BUILDING SELECTIVE INDEX: {source_name}")
    print("=" * 70)

    start = time.time()

    indexes = {
        family: CappedIndex(limit)
        for family, limit in LIMITS.items()
    }

    rows_seen = 0

    reader = pd.read_csv(
        filename,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:

        rows_seen += len(chunk)

        print(
            f"{source_name}: "
            f"{rows_seen:,} rows",
            end="\r",
        )

        for row in chunk.itertuples(
            index=False
        ):

            country = normalize_country(
                row.country
            )

            name = safe(
                row.business_name
            )

            address = safe(
                row.business_address
            )

            entity_id = safe(
                row.entity_id
            )

            basic = normalize_name(
                name
            )

            no_suffix = name_no_suffix(
                name
            )

            compact = re.sub(
                r"\s+",
                "",
                basic,
            )

            # ----------------------------
            # Exact name
            # ----------------------------

            if basic:

                indexes[
                    "exact_name"
                ].add(
                    f"{country}|{basic}",
                    entity_id,
                )

            # ----------------------------
            # Name no suffix
            # ----------------------------

            if no_suffix:

                indexes[
                    "name_no_suffix"
                ].add(
                    f"{country}|{no_suffix}",
                    entity_id,
                )

            # ----------------------------
            # Prefix5
            # ----------------------------

            if compact:

                indexes[
                    "prefix5"
                ].add(
                    f"{country}|{compact[:5]}",
                    entity_id,
                )

            # ----------------------------
            # Name tokens
            # ----------------------------

            for token in name_tokens(
                name
            ):

                indexes[
                    "name_token"
                ].add(
                    f"{country}|{token}",
                    entity_id,
                )

            # ----------------------------
            # Address number
            # ----------------------------

            number = first_number(
                address
            )

            if number:

                indexes[
                    "address_number"
                ].add(
                    f"{country}|{number}",
                    entity_id,
                )

            # ----------------------------
            # Postal
            # ----------------------------

            for postal in postal_codes(
                address
            ):

                indexes[
                    "postal"
                ].add(
                    f"{country}|{postal}",
                    entity_id,
                )

            # ----------------------------
            # Address tokens
            # ----------------------------

            for token in address_tokens(
                address
            ):

                indexes[
                    "address_token"
                ].add(
                    f"{country}|{token}",
                    entity_id,
                )

    print()

    elapsed = time.time() - start

    print(
        f"Finished {source_name} "
        f"in {elapsed:.1f}s"
    )

    # Statistics
    for family, index in indexes.items():

        total_keys = len(
            index.data
        )

        usable = sum(
            1
            for values in index.data.values()
            if values is not None
        )

        overflowed = (
            total_keys - usable
        )

        print(
            f"{family:20s} "
            f"keys={total_keys:,} "
            f"usable={usable:,} "
            f"overflowed={overflowed:,}"
        )

    return indexes


# ============================================================
# CANDIDATES FOR ONE RECORD
# ============================================================

def generate_candidates(
    row,
    index,
):
    """
    Generate candidates for one S1 record using the
    frequency-capped blocking indexes.
    """

    candidates = set()

    country = normalize_country(
        row.country
    )

    name = safe(
        row.business_name
    )

    address = safe(
        row.business_address
    )

    basic = normalize_name(
        name
    )

    no_suffix = name_no_suffix(
        name
    )

    compact = re.sub(
        r"\s+",
        "",
        basic,
    )

    # --------------------------------------------------------
    # Exact normalized name
    # --------------------------------------------------------

    if basic:

        candidates.update(
            index["exact_name"].get(
                f"{country}|{basic}"
            )
        )

    # --------------------------------------------------------
    # Name without legal suffix
    # --------------------------------------------------------

    if no_suffix:

        candidates.update(
            index["name_no_suffix"].get(
                f"{country}|{no_suffix}"
            )
        )

    # --------------------------------------------------------
    # Selective name prefix
    # --------------------------------------------------------

    if compact:

        candidates.update(
            index["prefix5"].get(
                f"{country}|{compact[:5]}"
            )
        )

    # --------------------------------------------------------
    # Informative name tokens
    # --------------------------------------------------------

    for token in name_tokens(name):

        candidates.update(
            index["name_token"].get(
                f"{country}|{token}"
            )
        )

    # --------------------------------------------------------
    # Address number
    # --------------------------------------------------------

    number = first_number(
        address
    )

    if number:

        candidates.update(
            index["address_number"].get(
                f"{country}|{number}"
            )
        )

    # --------------------------------------------------------
    # Postal code
    # --------------------------------------------------------

    for postal in postal_codes(address):

        candidates.update(
            index["postal"].get(
                f"{country}|{postal}"
            )
        )

    # --------------------------------------------------------
    # Selective address tokens
    # --------------------------------------------------------

    for token in address_tokens(address):

        candidates.update(
            index["address_token"].get(
                f"{country}|{token}"
            )
        )

    return candidates

# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    s1, truth = load_validation()

    print(
        f"\nValidation S1: "
        f"{len(s1):,}"
    )

    print(
        f"Known true matches: "
        f"{sum(len(x) for x in truth.values()):,}"
    )

    # --------------------------------------------------------
    # Build full-source selective indexes
    # --------------------------------------------------------

    s2_index = build_source_index(
        S2_FILE,
        "SOURCE2",
    )

    s3_index = build_source_index(
        S3_FILE,
        "SOURCE3",
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    total_true = 0
    total_found = 0

    candidate_counts = []

    missing = []

    for row in s1.itertuples(
        index=False
    ):

        s1_id = row.entity_id

        true_ids = truth.get(
            s1_id,
            set(),
        )

        s2_candidates = generate_candidates(
            row,
            s2_index,
        )

        s3_candidates = generate_candidates(
            row,
            s3_index,
        )

        candidates = (
            s2_candidates
            | s3_candidates
        )

        found = (
            true_ids
            & candidates
        )

        total_true += len(
            true_ids
        )

        total_found += len(
            found
        )

        candidate_counts.append(
            len(candidates)
        )

        if true_ids - candidates:

            for entity_id in (
                true_ids - candidates
            ):

                if len(missing) < 30:

                    missing.append(
                        (
                            s1_id,
                            entity_id,
                        )
                    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    recall = (
        total_found / total_true
        if total_true
        else 0
    )

    counts = pd.Series(
        candidate_counts
    )

    print()
    print("=" * 70)
    print("SELECTIVE FULL-SOURCE BLOCKING RESULTS")
    print("=" * 70)

    print(
        f"Validation S1:        {len(s1):,}"
    )

    print(
        f"True matches:          {total_true:,}"
    )

    print(
        f"Recovered:             {total_found:,}"
    )

    print(
        f"Blocking recall:       {recall:.4%}"
    )

    print(
        f"\nAverage candidates/S1: "
        f"{counts.mean():.2f}"
    )

    print(
        f"Median candidates/S1:  "
        f"{counts.median():.2f}"
    )

    print(
        f"95th percentile:       "
        f"{counts.quantile(.95):.2f}"
    )

    print(
        f"Maximum candidates:    "
        f"{counts.max():,.0f}"
    )

    if missing:

        print(
            "\nMissing true matches:"
        )

        for sid, mid in missing:

            print(
                f"{sid} -> {mid}"
            )

    else:

        print(
            "\nNo known true matches missed."
        )

    print(
        f"\nTotal runtime: "
        f"{time.time() - total_start:.1f}s"
    )


if __name__ == "__main__":
    main()