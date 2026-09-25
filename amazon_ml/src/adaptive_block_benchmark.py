from __future__ import annotations

import re
import time
from collections import Counter, defaultdict

import pandas as pd

from .normalize import (
    normalize_basic,
    normalize_compact,
    remove_legal_suffix,
    transliterate,
    extract_numbers,
)


# ============================================================
# PATHS
# ============================================================

BASE = (
    r"C:\Users\darsh\Downloads"
    r"\6ab10eb3b23ba_student_resource"
    r"\student_resource\dataset\train"
)

S2_FILE = BASE + r"\train_source2.tsv"
S3_FILE = BASE + r"\train_source3.tsv"

S1_FILE = r"diagnostics\sample_s1.tsv"
GT_FILE = r"diagnostics\sample_ground_truth.tsv"

CHUNK_SIZE = 250_000


# ============================================================
# SETTINGS
# ============================================================

# Strong deterministic blocks.
MAX_EXACT_BLOCK = 500

# Maximum document frequency for a token/gram to be considered
# a useful adaptive blocking key.
RARE_MAX_DF = 500

# Number of rare signals selected per S1.
RARE_NAME_TOKENS = 2
RARE_TRANSLIT_TOKENS = 2
RARE_ADDRESS_TOKENS = 2

RARE_NAME_NGRAMS = 2
RARE_TRANSLIT_NGRAMS = 2
RARE_ADDRESS_NGRAMS = 1

NGRAM_SIZE = 3


# ============================================================
# GENERIC TOKENS
# ============================================================

GENERIC_NAME = {
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

GENERIC_ADDRESS = {
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


WEBSITE_SUFFIXES = {
    "com",
    "net",
    "org",
    "co",
    "in",
    "biz",
    "info",
    "us",
}


# ============================================================
# HELPERS
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


def country(value) -> str:

    return safe(value).lower().strip()


def basic_name(value) -> str:

    return normalize_basic(
        safe(value)
    )


def no_suffix_name(value) -> str:

    return remove_legal_suffix(
        safe(value)
    )


def compact(value) -> str:

    return normalize_compact(
        safe(value)
    )


def name_tokens(value) -> list[str]:

    text = basic_name(value)

    if not text:
        return []

    return sorted({
        token
        for token in text.split()
        if len(token) >= 3
        and token not in GENERIC_NAME
    })


def translit_tokens(value) -> list[str]:

    text = transliterate(
        safe(value)
    )

    if not text:
        return []

    return sorted({
        token
        for token in text.split()
        if len(token) >= 3
        and token not in GENERIC_NAME
    })


def address_tokens(value) -> list[str]:

    text = basic_name(value)

    if not text:
        return []

    return sorted({
        token
        for token in text.split()
        if len(token) >= 4
        and token not in GENERIC_ADDRESS
    })


def number_key(value) -> str:

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
        x
        for x in extract_numbers(
            safe(value)
        )
        if len(x) in (5, 6)
    ]


def char_ngrams(
    value: str,
    n: int = NGRAM_SIZE,
) -> set[str]:

    value = safe(value)

    if len(value) < n:
        return set()

    return {
        value[i:i + n]
        for i in range(
            len(value) - n + 1
        )
    }


def name_ngrams(value) -> set[str]:

    text = compact(
        basic_name(value)
    )

    return char_ngrams(
        text
    )


def translit_name_ngrams(value) -> set[str]:

    text = re.sub(
        r"[^a-z0-9]",
        "",
        transliterate(
            safe(value)
        ),
    )

    return char_ngrams(
        text
    )


def address_ngrams(value) -> set[str]:

    grams = set()

    for token in address_tokens(
        value
    ):

        if len(token) >= 5:

            grams.update(
                char_ngrams(
                    token
                )
            )

    return grams


def domain_name(value) -> str:

    text = basic_name(
        value
    )

    if not text:
        return ""

    text = no_suffix_name(
        text
    )

    tokens = text.split()

    while (
        tokens
        and tokens[-1]
        in WEBSITE_SUFFIXES
    ):
        tokens.pop()

    return "".join(tokens)


# ============================================================
# QUERY SIGNATURE
# ============================================================

def signature(row) -> dict:

    name = safe(
        row.business_name
    )

    address = safe(
        row.business_address
    )

    return {
        "entity_id":
            safe(row.entity_id),

        "country":
            country(row.country),

        "name":
            name,

        "address":
            address,

        "basic_name":
            basic_name(name),

        "no_suffix":
            no_suffix_name(name),

        "translit_name":
            transliterate(name),

        "domain_name":
            domain_name(name),

        "name_tokens":
            name_tokens(name),

        "translit_tokens":
            translit_tokens(name),

        "address_tokens":
            address_tokens(address),

        "number":
            number_key(address),

        "postal":
            postal_codes(address),

        "name_ngrams":
            name_ngrams(name),

        "translit_ngrams":
            translit_name_ngrams(name),

        "address_ngrams":
            address_ngrams(address),
    }


# ============================================================
# VALIDATION DATA
# ============================================================

def load_validation():

    print(
        "Loading validation data..."
    )

    s1 = pd.read_csv(
        S1_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    gt = pd.read_csv(
        GT_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    signatures = {}

    for row in s1.itertuples(
        index=False
    ):

        sig = signature(row)

        signatures[
            sig["entity_id"]
        ] = sig

    truth = {}

    for row in gt.itertuples(
        index=False
    ):

        value = safe(
            row.matched_entity_ids
        )

        truth[
            row.source1_entity_id
        ] = (
            {
                x.strip()
                for x in value.split(",")
                if x.strip()
            }
            if value
            else set()
        )

    return signatures, truth


# ============================================================
# BUILD QUERY UNIVERSE
# ============================================================

def build_query_universe(
    signatures
):

    universe = {
        "name_tokens": set(),
        "translit_tokens": set(),
        "address_tokens": set(),
        "name_ngrams": set(),
        "translit_ngrams": set(),
        "address_ngrams": set(),

        "exact_name": set(),
        "no_suffix": set(),
        "translit_name": set(),
        "domain_name": set(),
        "postal": set(),
        "name_number": set(),
    }

    for sig in signatures.values():

        c = sig["country"]

        # Exact keys
        if sig["basic_name"]:

            universe[
                "exact_name"
            ].add(
                f"{c}|{sig['basic_name']}"
            )

        if sig["no_suffix"]:

            universe[
                "no_suffix"
            ].add(
                f"{c}|{sig['no_suffix']}"
            )

        if sig["translit_name"]:

            universe[
                "translit_name"
            ].add(
                f"{c}|{sig['translit_name']}"
            )

        if sig["domain_name"]:

            universe[
                "domain_name"
            ].add(
                f"{c}|{sig['domain_name']}"
            )

        # Tokens
        for token in sig[
            "name_tokens"
        ]:

            universe[
                "name_tokens"
            ].add(
                f"{c}|{token}"
            )

        for token in sig[
            "translit_tokens"
        ]:

            universe[
                "translit_tokens"
            ].add(
                f"{c}|{token}"
            )

        for token in sig[
            "address_tokens"
        ]:

            universe[
                "address_tokens"
            ].add(
                f"{c}|{token}"
            )

        # Ngrams
        for gram in sig[
            "name_ngrams"
        ]:

            universe[
                "name_ngrams"
            ].add(
                f"{c}|{gram}"
            )

        for gram in sig[
            "translit_ngrams"
        ]:

            universe[
                "translit_ngrams"
            ].add(
                f"{c}|{gram}"
            )

        for gram in sig[
            "address_ngrams"
        ]:

            universe[
                "address_ngrams"
            ].add(
                f"{c}|{gram}"
            )

    return universe


# ============================================================
# COUNT DOCUMENT FREQUENCIES
# ============================================================

def count_frequencies(
    filename,
    source_name,
    universe,
):

    print()
    print(
        "=" * 70
    )
    print(
        f"COUNTING RARE KEYS: {source_name}"
    )
    print(
        "=" * 70
    )

    counters = {
        family: Counter()
        for family in [
            "name_tokens",
            "translit_tokens",
            "address_tokens",
            "name_ngrams",
            "translit_ngrams",
            "address_ngrams",
            "exact_name",
            "no_suffix",
            "translit_name",
            "domain_name",
        ]
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

            c = country(
                row.country
            )

            sig = signature(row)

            # Each record counts ONCE per key.
            # This is document frequency.

            for token in sig[
                "name_tokens"
            ]:

                key = f"{c}|{token}"

                if key in universe[
                    "name_tokens"
                ]:

                    counters[
                        "name_tokens"
                    ][key] += 1

            for token in sig[
                "translit_tokens"
            ]:

                key = f"{c}|{token}"

                if key in universe[
                    "translit_tokens"
                ]:

                    counters[
                        "translit_tokens"
                    ][key] += 1

            for token in sig[
                "address_tokens"
            ]:

                key = f"{c}|{token}"

                if key in universe[
                    "address_tokens"
                ]:

                    counters[
                        "address_tokens"
                    ][key] += 1

            for gram in sig[
                "name_ngrams"
            ]:

                key = f"{c}|{gram}"

                if key in universe[
                    "name_ngrams"
                ]:

                    counters[
                        "name_ngrams"
                    ][key] += 1

            for gram in sig[
                "translit_ngrams"
            ]:

                key = f"{c}|{gram}"

                if key in universe[
                    "translit_ngrams"
                ]:

                    counters[
                        "translit_ngrams"
                    ][key] += 1

            for gram in sig[
                "address_ngrams"
            ]:

                key = f"{c}|{gram}"

                if key in universe[
                    "address_ngrams"
                ]:

                    counters[
                        "address_ngrams"
                    ][key] += 1

            # Exact key frequencies are useful for diagnostics.
            if (
                sig["basic_name"]
                and
                f"{c}|{sig['basic_name']}"
                in universe["exact_name"]
            ):

                counters[
                    "exact_name"
                ][
                    f"{c}|{sig['basic_name']}"
                ] += 1

            if (
                sig["no_suffix"]
                and
                f"{c}|{sig['no_suffix']}"
                in universe["no_suffix"]
            ):

                counters[
                    "no_suffix"
                ][
                    f"{c}|{sig['no_suffix']}"
                ] += 1

            if (
                sig["translit_name"]
                and
                f"{c}|{sig['translit_name']}"
                in universe["translit_name"]
            ):

                counters[
                    "translit_name"
                ][
                    f"{c}|{sig['translit_name']}"
                ] += 1

            if (
                sig["domain_name"]
                and
                f"{c}|{sig['domain_name']}"
                in universe["domain_name"]
            ):

                counters[
                    "domain_name"
                ][
                    f"{c}|{sig['domain_name']}"
                ] += 1

    print()

    return counters


# ============================================================
# CHOOSE RARE QUERY KEYS
# ============================================================

def choose_rare_keys(
    signatures,
    counters,
    family,
    count_per_query,
):

    selected = defaultdict(set)

    for sid, sig in signatures.items():

        c = sig["country"]

        if family == "name_tokens":
            values = sig[
                "name_tokens"
            ]

        elif family == "translit_tokens":
            values = sig[
                "translit_tokens"
            ]

        elif family == "address_tokens":
            values = sig[
                "address_tokens"
            ]

        elif family == "name_ngrams":
            values = sig[
                "name_ngrams"
            ]

        elif family == "translit_ngrams":
            values = sig[
                "translit_ngrams"
            ]

        elif family == "address_ngrams":
            values = sig[
                "address_ngrams"
            ]

        else:
            raise ValueError(
                f"Unknown family: {family}"
            )

        scored = []

        for value in values:

            key = f"{c}|{value}"

            df = counters[
                family
            ].get(
                key,
                0,
            )

            if 0 < df <= RARE_MAX_DF:

                scored.append(
                    (
                        df,
                        key,
                    )
                )

        scored.sort(
            key=lambda x: x[0]
        )

        for _, key in scored[
            :count_per_query
        ]:

            selected[sid].add(
                key
            )

    return selected


# ============================================================
# BUILD SECOND-PASS POSTINGS
# ============================================================

def build_postings(
    filename,
    source_name,
    signatures,
    selected_keys,
    universe,
):

    print()
    print(
        "=" * 70
    )
    print(
        f"BUILDING SELECTIVE POSTINGS: "
        f"{source_name}"
    )
    print(
        "=" * 70
    )

    postings = {
        family: defaultdict(list)
        for family in [
            "exact_name",
            "no_suffix",
            "translit_name",
            "domain_name",
            "name_tokens",
            "translit_tokens",
            "address_tokens",
            "name_ngrams",
            "translit_ngrams",
            "address_ngrams",
        ]
    }

    rows_seen = 0

    # Determine all selected keys once.
    selected_global = defaultdict(set)

    for family, per_sid in selected_keys.items():

        for sid, keys in per_sid.items():

            selected_global[
                family
            ].update(keys)

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

            entity_id = safe(
                row.entity_id
            )

            sig = signature(row)

            c = sig["country"]

            # ------------------------------------------------
            # Strong exact blocks
            # ------------------------------------------------

            if sig["basic_name"]:

                key = (
                    f"{c}|"
                    f"{sig['basic_name']}"
                )

                if key in universe[
                    "exact_name"
                ]:

                    postings[
                        "exact_name"
                    ][key].append(
                        entity_id
                    )

            if sig["no_suffix"]:

                key = (
                    f"{c}|"
                    f"{sig['no_suffix']}"
                )

                if key in universe[
                    "no_suffix"
                ]:

                    postings[
                        "no_suffix"
                    ][key].append(
                        entity_id
                    )

            if sig["translit_name"]:

                key = (
                    f"{c}|"
                    f"{sig['translit_name']}"
                )

                if key in universe[
                    "translit_name"
                ]:

                    postings[
                        "translit_name"
                    ][key].append(
                        entity_id
                    )

            if sig["domain_name"]:

                key = (
                    f"{c}|"
                    f"{sig['domain_name']}"
                )

                if key in universe[
                    "domain_name"
                ]:

                    postings[
                        "domain_name"
                    ][key].append(
                        entity_id
                    )

            # ------------------------------------------------
            # Adaptive rare blocks
            # ------------------------------------------------

            families = [
                "name_tokens",
                "translit_tokens",
                "address_tokens",
                "name_ngrams",
                "translit_ngrams",
                "address_ngrams",
            ]

            values_map = {
                "name_tokens":
                    sig["name_tokens"],

                "translit_tokens":
                    sig["translit_tokens"],

                "address_tokens":
                    sig["address_tokens"],

                "name_ngrams":
                    sig["name_ngrams"],

                "translit_ngrams":
                    sig["translit_ngrams"],

                "address_ngrams":
                    sig["address_ngrams"],
            }

            for family in families:

                values = values_map[
                    family
                ]

                if not values:
                    continue

                selected_for_family = (
                    selected_global[
                        family
                    ]
                )

                for value in values:

                    key = (
                        f"{c}|{value}"
                    )

                    if key in selected_for_family:

                        postings[
                            family
                        ][key].append(
                            entity_id
                        )

    print()

    return postings


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def candidates_for_s1(
    sid,
    sig,
    postings,
    selected_keys,
):

    result = set()

    c = sig["country"]

    # --------------------------------------------------------
    # Strong exact blocks
    # --------------------------------------------------------

    strong = [
        (
            "exact_name",
            sig["basic_name"],
        ),
        (
            "no_suffix",
            sig["no_suffix"],
        ),
        (
            "translit_name",
            sig["translit_name"],
        ),
        (
            "domain_name",
            sig["domain_name"],
        ),
    ]

    for family, value in strong:

        if not value:
            continue

        key = f"{c}|{value}"

        values = postings[
            family
        ].get(
            key,
            []
        )

        if len(values) <= MAX_EXACT_BLOCK:

            result.update(
                values
            )

    # --------------------------------------------------------
    # Rare blocks selected specifically for S1
    # --------------------------------------------------------

    for family, per_sid in (
        selected_keys.items()
    ):

        keys = per_sid.get(
            sid,
            set()
        )

        for key in keys:

            values = postings[
                family
            ].get(
                key,
                []
            )

            result.update(
                values
            )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    signatures, truth = (
        load_validation()
    )

    print(
        f"\nValidation S1: "
        f"{len(signatures):,}"
    )

    print(
        f"True matches: "
        f"{sum(len(x) for x in truth.values()):,}"
    )

    # --------------------------------------------------------
    # Query universe
    # --------------------------------------------------------

    print(
        "\nBuilding query universe..."
    )

    universe = build_query_universe(
        signatures
    )

    for family, values in universe.items():

        print(
            f"{family:20s}: "
            f"{len(values):,}"
        )

    # --------------------------------------------------------
    # First pass: frequencies
    # --------------------------------------------------------

    freq_s2 = count_frequencies(
        S2_FILE,
        "SOURCE2",
        universe,
    )

    freq_s3 = count_frequencies(
        S3_FILE,
        "SOURCE3",
        universe,
    )

    # --------------------------------------------------------
    # Select rare keys separately for
    # S2 and S3
    # --------------------------------------------------------

    selected_s2 = {}
    selected_s3 = {}

    for family, n in [
        (
            "name_tokens",
            RARE_NAME_TOKENS,
        ),
        (
            "translit_tokens",
            RARE_TRANSLIT_TOKENS,
        ),
        (
            "address_tokens",
            RARE_ADDRESS_TOKENS,
        ),
        (
            "name_ngrams",
            RARE_NAME_NGRAMS,
        ),
        (
            "translit_ngrams",
            RARE_TRANSLIT_NGRAMS,
        ),
        (
            "address_ngrams",
            RARE_ADDRESS_NGRAMS,
        ),
    ]:

        selected_s2[
            family
        ] = choose_rare_keys(
            signatures,
            freq_s2,
            family,
            n,
        )

        selected_s3[
            family
        ] = choose_rare_keys(
            signatures,
            freq_s3,
            family,
            n,
        )

    # --------------------------------------------------------
    # Second pass: postings
    # --------------------------------------------------------

    postings_s2 = build_postings(
        S2_FILE,
        "SOURCE2",
        signatures,
        selected_s2,
        universe,
    )

    postings_s3 = build_postings(
        S3_FILE,
        "SOURCE3",
        signatures,
        selected_s3,
        universe,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    total_true = 0
    total_found = 0

    candidate_counts = []

    missing = []

    for sid, true_ids in truth.items():

        sig = signatures[sid]

        s2 = candidates_for_s1(
            sid,
            sig,
            postings_s2,
            selected_s2,
        )

        s3 = candidates_for_s1(
            sid,
            sig,
            postings_s3,
            selected_s3,
        )

        candidates = (
            s2 | s3
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

        for missed in (
            true_ids - candidates
        ):

            if len(missing) < 30:

                missing.append(
                    (
                        sid,
                        missed,
                    )
                )

    recall = (
        total_found / total_true
        if total_true
        else 0
    )

    counts = pd.Series(
        candidate_counts
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "ADAPTIVE V5 BLOCKING RESULTS"
    )
    print(
        "=" * 70
    )

    print(
        f"Validation S1:        "
        f"{len(truth):,}"
    )

    print(
        f"True matches:          "
        f"{total_true:,}"
    )

    print(
        f"Recovered:             "
        f"{total_found:,}"
    )

    print(
        f"Blocking recall:       "
        f"{recall:.4%}"
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

    # --------------------------------------------------------
    # Selected-key statistics
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "SELECTED RARE KEY COUNTS"
    )
    print(
        "=" * 70
    )

    for family in [
        "name_tokens",
        "translit_tokens",
        "address_tokens",
        "name_ngrams",
        "translit_ngrams",
        "address_ngrams",
    ]:

        s2_count = sum(
            len(v)
            for v in selected_s2[
                family
            ].values()
        )

        s3_count = sum(
            len(v)
            for v in selected_s3[
                family
            ].values()
        )

        print(
            f"{family:20s} "
            f"S2={s2_count:,} "
            f"S3={s3_count:,}"
        )

    print(
        f"\nTotal runtime: "
        f"{time.time() - start:.1f}s"
    )


if __name__ == "__main__":
    main()