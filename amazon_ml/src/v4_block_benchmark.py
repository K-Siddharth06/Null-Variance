from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from .normalize import (
    normalize_basic,
    remove_legal_suffix,
    extract_numbers,
    transliterate,
)


# ============================================================
# FULL DATASET
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
# BLOCK LIMITS
# ============================================================

LIMITS = {
    # Very strong exact-style blocks
    "exact_name": 500,
    "name_no_suffix": 500,

    # Cross-script / website-style exact blocks
    "translit_name": 500,
    "translit_no_suffix": 500,
    "domain_name": 500,

    # Strong composite blocks
    "name_postal": 20,
    "name_number": 150,
    "name_address": 150,
    "name_pair": 200,

    # Controlled address composites
    "address_pair": 200,
    "address_number": 75,
}


# ============================================================
# TOKENS
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


# ============================================================
# SAFE FUNCTIONS
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


def norm_country(value) -> str:
    return safe(value).lower().strip()


def norm_name(value) -> str:
    return normalize_basic(
        safe(value)
    )


def norm_no_suffix(value) -> str:
    return remove_legal_suffix(
        safe(value)
    )


def name_tokens(value) -> list[str]:

    text = norm_name(value)

    if not text:
        return []

    return sorted({
        token
        for token in text.split()
        if len(token) >= 3
        and token not in GENERIC_NAME
    })


def address_tokens(value) -> list[str]:

    text = norm_name(value)

    if not text:
        return []

    return sorted({
        token
        for token in text.split()
        if len(token) >= 4
        and token not in GENERIC_ADDRESS
    })


def numbers(value) -> list[str]:

    return extract_numbers(
        safe(value)
    )


def first_number(value) -> str:

    values = numbers(value)

    if not values:
        return ""

    for value in values:

        if len(value) <= 5:
            return value

    return values[0]


def postal_codes(value) -> list[str]:

    return [
        value
        for value in numbers(value)
        if len(value) in (5, 6)
    ]
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


def clean_domain_name(value: str) -> str:
    """
    Normalize website-style business names.

    Examples:

        Surjit Technologies Pvt Ltd
            -> surjittechnologies

        Surjittechnologies.Com
            -> surjittechnologies
    """

    text = normalize_basic(
        safe(value)
    )

    if not text:
        return ""

    text = remove_legal_suffix(
        text
    )

    tokens = text.split()

    while tokens and tokens[-1] in WEBSITE_SUFFIXES:
        tokens.pop()

    text = "".join(tokens)

    return text

# ============================================================
# KEY GENERATION
# ============================================================

def generate_keys(
    country,
    name,
    address,
) -> dict[str, list[str]]:

    country = norm_country(country)

    basic = norm_name(name)
    no_suffix = norm_no_suffix(name)

    translit_name = transliterate(name)
    translit_no_suffix = transliterate(no_suffix)
    domain_name = clean_domain_name(name)

    nt = name_tokens(name)
    at = address_tokens(address)

    number = first_number(address)
    postal = postal_codes(address)

    result = defaultdict(list)

    # --------------------------------------------------------
    # Exact name
    # --------------------------------------------------------

    if basic:

        result["exact_name"].append(
            f"{country}|{basic}"
        )

    # --------------------------------------------------------
    # Suffix stripped
    # --------------------------------------------------------

    if no_suffix:

        result["name_no_suffix"].append(
            f"{country}|{no_suffix}"
        )

    # --------------------------------------------------------
    # Exact transliteration
    # --------------------------------------------------------

    if translit_name:

        result["translit_name"].append(
            f"{country}|{translit_name}"
        )

    # --------------------------------------------------------
    # Transliteration without legal suffix
    # --------------------------------------------------------

    if translit_no_suffix:

        result["translit_no_suffix"].append(
            f"{country}|{translit_no_suffix}"
        )

    # --------------------------------------------------------
    # Website/domain-style compact name
    # --------------------------------------------------------

    if domain_name:

        result["domain_name"].append(
            f"{country}|{domain_name}"
        )

    # --------------------------------------------------------
    # Name + postal
    # --------------------------------------------------------

    for p in postal:

        for n in nt:

            result["name_postal"].append(
                f"{country}|{n}|{p}"
            )

    # --------------------------------------------------------
    # Name + number
    # --------------------------------------------------------

    if number:

        for n in nt:

            result["name_number"].append(
                f"{country}|{n}|{number}"
            )

    # --------------------------------------------------------
    # Name + address token
    # --------------------------------------------------------

    for n in nt:

        for a in at:

            result["name_address"].append(
                f"{country}|{n}|{a}"
            )

    # --------------------------------------------------------
    # Two name tokens
    # --------------------------------------------------------

    for i in range(len(nt)):

        for j in range(i + 1, len(nt)):

            result["name_pair"].append(
                f"{country}|{nt[i]}|{nt[j]}"
            )

    # --------------------------------------------------------
    # Two address tokens
    # --------------------------------------------------------

    for i in range(len(at)):

        for j in range(i + 1, len(at)):

            result["address_pair"].append(
                f"{country}|{at[i]}|{at[j]}"
            )

    # --------------------------------------------------------
    # Address token + number
    # --------------------------------------------------------

    if number:

        for a in at:

            result["address_number"].append(
                f"{country}|{number}|{a}"
            )

    return result


# ============================================================
# CAPPED POSTING LIST
# ============================================================

@dataclass
class Posting:

    limit: int

    values: list[str] | None = None

    overflowed: bool = False

    def add(self, entity_id: str):

        if self.overflowed:
            return

        if self.values is None:

            self.values = [entity_id]

            return

        if len(self.values) >= self.limit:

            self.values = None
            self.overflowed = True

            return

        self.values.append(
            entity_id
        )


# ============================================================
# QUERY INDEX
# ============================================================

def load_validation():

    print("Loading validation data...")

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

    return s1, truth


def build_query_maps(s1):

    # family → key → S1 IDs
    query_map = {
        family: defaultdict(set)
        for family in LIMITS
    }

    for row in s1.itertuples(
        index=False
    ):

        keys = generate_keys(
            row.country,
            row.business_name,
            row.business_address,
        )

        for family, family_keys in keys.items():

            for key in family_keys:

                query_map[
                    family
                ][key].add(
                    row.entity_id
                )

    return query_map


# ============================================================
# FULL-SOURCE INDEX FOR VALIDATION QUERIES
# ============================================================

def scan_source(
    filename,
    source_name,
    query_map,
):

    print()
    print("=" * 70)
    print(f"SCANNING {source_name}")
    print("=" * 70)

    start = time.time()

    # family → query key → capped posting
    postings = {
        family: {}
        for family in LIMITS
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

            keys = generate_keys(
                row.country,
                row.business_name,
                row.business_address,
            )

            entity_id = row.entity_id

            for family, family_keys in keys.items():

                qmap = query_map[family]

                postings_family = postings[family]

                for key in family_keys:

                    # Only process keys that one of our
                    # validation S1 records actually has.
                    if key not in qmap:
                        continue

                    posting = postings_family.get(
                        key
                    )

                    if posting is None:

                        posting = Posting(
                            limit=LIMITS[family]
                        )

                        postings_family[key] = posting

                    posting.add(
                        entity_id
                    )

    elapsed = time.time() - start

    print()

    print(
        f"Finished {source_name} "
        f"in {elapsed:.1f}s"
    )

    return postings


# ============================================================
# TURN POSTINGS INTO S1 CANDIDATES
# ============================================================

def generate_candidates(
    query_map,
    postings,
):

    candidates = defaultdict(set)

    for family in LIMITS:

        qmap = query_map[family]

        for key, s1_ids in qmap.items():

            posting = postings[
                family
            ].get(key)

            if posting is None:
                continue

            if posting.overflowed:
                continue

            if not posting.values:
                continue

            for s1_id in s1_ids:

                candidates[
                    s1_id
                ].update(
                    posting.values
                )

    return candidates


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()

    s1, truth = load_validation()

    print(
        f"S1 validation records: "
        f"{len(s1):,}"
    )

    print(
        f"True matches: "
        f"{sum(len(x) for x in truth.values()):,}"
    )

    print("\nBuilding query maps...")

    query_map = build_query_maps(
        s1
    )

    for family in LIMITS:

        print(
            f"{family:20s}: "
            f"{len(query_map[family]):,} query keys"
        )

    # --------------------------------------------------------
    # Full Source 2
    # --------------------------------------------------------

    postings_s2 = scan_source(
        S2_FILE,
        "SOURCE2",
        query_map,
    )

    # --------------------------------------------------------
    # Full Source 3
    # --------------------------------------------------------

    postings_s3 = scan_source(
        S3_FILE,
        "SOURCE3",
        query_map,
    )

    # --------------------------------------------------------
    # Candidates
    # --------------------------------------------------------

    candidates_s2 = generate_candidates(
        query_map,
        postings_s2,
    )

    candidates_s3 = generate_candidates(
        query_map,
        postings_s3,
    )

    all_candidates = defaultdict(set)

    for sid, values in candidates_s2.items():

        all_candidates[
            sid
        ].update(
            values
        )

    for sid, values in candidates_s3.items():

        all_candidates[
            sid
        ].update(
            values
        )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_true = 0
    total_found = 0

    candidate_counts = []

    missing = []

    for sid, true_ids in truth.items():

        predicted = (
            all_candidates.get(
                sid,
                set()
            )
        )

        found = true_ids & predicted

        total_true += len(
            true_ids
        )

        total_found += len(
            found
        )

        candidate_counts.append(
            len(predicted)
        )

        not_found = (
            true_ids - predicted
        )

        for mid in not_found:

            if len(missing) < 30:

                missing.append(
                    (sid, mid)
                )

    recall = (
        total_found / total_true
        if total_true
        else 0.0
    )

    counts = pd.Series(
        candidate_counts
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("V4.1 COMPOSITE BLOCKING RESULTS")
    print("=" * 70)

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

    # --------------------------------------------------------
    # Missing positives
    # --------------------------------------------------------

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
    # Posting statistics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("POSTING OVERFLOW COUNTS")
    print("=" * 70)

    for family in LIMITS:

        s2_overflow = sum(
            1
            for x in postings_s2[
                family
            ].values()
            if x.overflowed
        )

        s3_overflow = sum(
            1
            for x in postings_s3[
                family
            ].values()
            if x.overflowed
        )

        print(
            f"{family:20s} "
            f"S2 overflow={s2_overflow:,} "
            f"S3 overflow={s3_overflow:,}"
        )

    print(
        f"\nTotal runtime: "
        f"{time.time() - overall_start:.1f}s"
    )


if __name__ == "__main__":
    main()