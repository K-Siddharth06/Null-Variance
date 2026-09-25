from __future__ import annotations

import time
import re
from collections import defaultdict

import pandas as pd


# ============================================================
# FULL DATASET PATHS
# ============================================================

BASE = (
    r"C:\Users\darsh\Downloads"
    r"\6ab10eb3b23ba_student_resource"
    r"\student_resource\dataset\train"
)

S1_FILE = BASE + r"\train_source1.tsv"
S2_FILE = BASE + r"\train_source2.tsv"
S3_FILE = BASE + r"\train_source3.tsv"

SAMPLE_S1 = r"diagnostics\sample_s1.tsv"
SAMPLE_GT = r"diagnostics\sample_ground_truth.tsv"

CHUNK_SIZE = 250_000


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


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_basic_series(s: pd.Series) -> pd.Series:
    """
    Fast vectorized normalization for the scale benchmark.

    This is deliberately simpler than the full normalize.py
    implementation so we can process millions of rows quickly.
    """

    s = (
        s.fillna("")
        .astype(str)
        .str.lower()
        .str.normalize("NFC")
    )

    s = s.str.replace("&", " and ", regex=False)

    s = s.str.replace(
        r"[^\w\s]",
        " ",
        regex=True,
    )

    s = s.str.replace(
        r"\s+",
        " ",
        regex=True,
    )

    return s.str.strip()


def compact_series(s: pd.Series) -> pd.Series:

    return (
        s.str.replace(
            r"\s+",
            "",
            regex=True,
        )
    )


def remove_suffix_series(s: pd.Series) -> pd.Series:

    suffix_pattern = (
        r"(?:\s+"
        r"(?:private limited|pvt limited|pvt ltd|"
        r"private ltd|limited|ltd|corporation|corp|"
        r"incorporated|inc|llc|pllc|co|company))+$"
    )

    return s.str.replace(
        suffix_pattern,
        "",
        regex=True,
    ).str.strip()


def first_informative_token(value: str) -> str:

    if not value:
        return ""

    for token in value.split():

        if len(token) < 3:
            continue

        if token in GENERIC_NAME_TOKENS:
            continue

        return token

    return ""


def first_number(value: str) -> str:

    if not value:
        return ""

    numbers = re.findall(
        r"\d+",
        value,
    )

    if not numbers:
        return ""

    for n in numbers:

        if len(n) <= 5:
            return n

    return numbers[0]


def postal_codes(value: str) -> list[str]:

    if not value:
        return []

    return [
        x
        for x in re.findall(
            r"\d+",
            value,
        )
        if len(x) in (5, 6)
    ]


# ============================================================
# BUILD QUERY INDEX
# ============================================================

def build_query_index():

    print("Loading validation S1...")

    s1 = pd.read_csv(
        SAMPLE_S1,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    gt = pd.read_csv(
        SAMPLE_GT,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s1["name_basic"] = normalize_basic_series(
        s1["business_name"]
    )

    s1["name_compact"] = compact_series(
        s1["name_basic"]
    )

    s1["name_no_suffix"] = remove_suffix_series(
        s1["name_basic"]
    )

    s1["name_prefix5"] = (
        s1["name_compact"]
        .str[:5]
    )

    s1["first_token"] = (
        s1["name_basic"]
        .map(first_informative_token)
    )

    s1["address_normalized"] = (
        normalize_basic_series(
            s1["business_address"]
        )
    )

    s1["address_number"] = (
        s1["address_normalized"]
        .map(first_number)
    )

    # --------------------------------------------------------
    # Reverse lookup:
    #
    # blocking key → Source-1 IDs
    # --------------------------------------------------------

    indexes = {
        "exact_name": defaultdict(set),
        "name_no_suffix": defaultdict(set),
        "name_prefix5": defaultdict(set),
        "first_token": defaultdict(set),
        "address_number": defaultdict(set),
        "postal": defaultdict(set),
    }

    for row in s1.itertuples(index=False):

        sid = row.entity_id
        country = str(row.country).lower().strip()

        if row.name_basic:
            indexes["exact_name"][
                (country, row.name_basic)
            ].add(sid)

        if row.name_no_suffix:
            indexes["name_no_suffix"][
                (country, row.name_no_suffix)
            ].add(sid)

        if row.name_prefix5:
            indexes["name_prefix5"][
                (country, row.name_prefix5)
            ].add(sid)

        if row.first_token:
            indexes["first_token"][
                (country, row.first_token)
            ].add(sid)

        if row.address_number:
            indexes["address_number"][
                (country, row.address_number)
            ].add(sid)

        for postal in postal_codes(
            row.address_normalized
        ):
            indexes["postal"][
                (country, postal)
            ].add(sid)

    # --------------------------------------------------------
    # Truth
    # --------------------------------------------------------

    truth = {}

    for row in gt.itertuples(index=False):

        if not row.matched_entity_ids:

            truth[row.source1_entity_id] = set()

        else:

            truth[
                row.source1_entity_id
            ] = {
                x.strip()
                for x in row.matched_entity_ids.split(",")
                if x.strip()
            }

    return s1, truth, indexes


# ============================================================
# SCAN ONE SOURCE
# ============================================================

def scan_source(
    filename: str,
    indexes,
    source_name: str,
):

    print("\n" + "=" * 70)
    print(f"Scanning {source_name}")
    print("=" * 70)

    start = time.time()

    candidates = defaultdict(set)

    rows_seen = 0

    reader = pd.read_csv(
        filename,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    )

    for chunk_no, chunk in enumerate(
        reader,
        start=1,
    ):

        rows_seen += len(chunk)

        print(
            f"{source_name}: "
            f"{rows_seen:,} rows scanned",
            end="\r",
        )

        country = (
            chunk["country"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

        name = normalize_basic_series(
            chunk["business_name"]
        )

        compact = compact_series(name)

        no_suffix = remove_suffix_series(
            name
        )

        prefix5 = compact.str[:5]

        first_token = name.map(
            first_informative_token
        )

        address = normalize_basic_series(
            chunk["business_address"]
        )

        address_number = address.map(
            first_number
        )

        entity_ids = (
            chunk["entity_id"]
            .astype(str)
            .tolist()
        )

        countries = country.tolist()
        names = name.tolist()
        no_suffixes = no_suffix.tolist()
        prefixes = prefix5.tolist()
        first_tokens = first_token.tolist()
        numbers = address_number.tolist()
        addresses = address.tolist()

        for i, entity_id in enumerate(
            entity_ids
        ):

            c = countries[i]

            keys = [
                (
                    "exact_name",
                    (c, names[i]),
                ),
                (
                    "name_no_suffix",
                    (c, no_suffixes[i]),
                ),
                (
                    "name_prefix5",
                    (c, prefixes[i]),
                ),
                (
                    "first_token",
                    (c, first_tokens[i]),
                ),
                (
                    "address_number",
                    (c, numbers[i]),
                ),
            ]

            # Postal blocks
            for postal in postal_codes(
                addresses[i]
            ):

                keys.append(
                    (
                        "postal",
                        (c, postal),
                    )
                )

            for index_name, key in keys:

                if not key[1]:
                    continue

                matched_s1 = indexes[
                    index_name
                ].get(key)

                if not matched_s1:
                    continue

                for s1_id in matched_s1:

                    candidates[
                        s1_id
                    ].add(entity_id)

    elapsed = time.time() - start

    print()
    print(
        f"Finished {source_name} "
        f"in {elapsed:.1f} seconds"
    )

    return candidates


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    s1, truth, indexes = (
        build_query_index()
    )

    print(
        f"\nValidation S1: "
        f"{len(s1):,}"
    )

    print(
        f"Known true matches: "
        f"{sum(len(x) for x in truth.values()):,}"
    )

    s2_candidates = scan_source(
        S2_FILE,
        indexes,
        "SOURCE 2",
    )

    s3_candidates = scan_source(
        S3_FILE,
        indexes,
        "SOURCE 3",
    )

    # --------------------------------------------------------
    # Merge S2 + S3 candidates
    # --------------------------------------------------------

    all_candidates = defaultdict(set)

    for sid, ids in s2_candidates.items():
        all_candidates[sid].update(ids)

    for sid, ids in s3_candidates.items():
        all_candidates[sid].update(ids)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_true = 0
    total_found = 0

    candidate_counts = []

    missing_examples = []

    for sid, true_ids in truth.items():

        candidates = all_candidates.get(
            sid,
            set(),
        )

        found = true_ids & candidates

        total_true += len(true_ids)
        total_found += len(found)

        candidate_counts.append(
            len(candidates)
        )

        missing = true_ids - candidates

        for mid in missing:

            if len(missing_examples) < 30:

                missing_examples.append(
                    (
                        sid,
                        mid,
                    )
                )

    recall = (
        total_found / total_true
        if total_true
        else 0.0
    )

    counts = pd.Series(
        candidate_counts
    )

    print("\n")
    print("=" * 70)
    print("REAL-SCALE BLOCKING RESULTS")
    print("=" * 70)

    print(
        f"Validation S1:        {len(truth):,}"
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

    if missing_examples:

        print(
            "\nMissing true matches:"
        )

        for sid, mid in (
            missing_examples
        ):

            print(
                f"{sid} -> {mid}"
            )

    else:

        print(
            "\nNo known true matches missed."
        )

    print(
        f"\nTotal runtime: "
        f"{time.time() - total_start:.1f} seconds"
    )


if __name__ == "__main__":
    main()