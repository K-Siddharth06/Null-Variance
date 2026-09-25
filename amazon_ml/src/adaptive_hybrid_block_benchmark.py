from __future__ import annotations

import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from .normalize import (
    normalize_basic,
    normalize_compact,
    remove_legal_suffix,
    transliterate,
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

S1_FILE = r"diagnostics\sample_s1.tsv"
GT_FILE = r"diagnostics\sample_ground_truth.tsv"

CHUNK_SIZE = 250_000


# ============================================================
# STRONG BLOCK LIMITS
# ============================================================

# These are applied to individual source-side posting lists.
STRONG_LIMITS = {
    "exact_name": 500,
    "name_no_suffix": 500,
    "translit_name": 500,
    "domain_name": 500,
    "name_postal": 20,
    "name_number": 150,
    "name_address": 150,
    "name_pair": 200,
}


# Rescue keys are first frequency-profiled on the full source.
# A key with document frequency > this is never stored as a rescue key.
MAX_RESCUE_DF = 100

# For difficult S1 records, take only the rarest few signals
# from each rescue family. This bounds the candidate explosion.
RAREST_NAME_TOKENS = 2
RAREST_TRANSLIT_TOKENS = 2
RAREST_ADDRESS_TOKENS = 2
RAREST_ADDRESS_PAIRS = 2
RAREST_ADDRESS_NUMBERS = 1

# Rescue is applied only when strong blocks are insufficient.
HARD_CANDIDATE_THRESHOLD = 20

# Candidate-count alternatives are reported from the same postings,
# so we do not need to rescan the 5M+ sources for every setting.
RESCUE_DF_THRESHOLDS = [10, 25, 50, 100]


# ============================================================
# TOKENS
# ============================================================

GENERIC_NAME = {
    "the", "and", "company", "corporation", "corporate", "limited",
    "private", "pvt", "ltd", "llc", "inc", "incorporated", "co",
    "group", "services", "service", "solutions", "solution",
    "enterprises", "enterprise", "business", "businesses",
    "international", "india", "america", "american", "usa", "united",
    "global", "official", "shop", "store", "mr", "mrs", "ms", "dr",
    "sri", "shri", "smt",
}

GENERIC_ADDRESS = {
    "road", "street", "st", "rd", "avenue", "ave", "lane", "ln",
    "drive", "dr", "highway", "hwy", "circle", "place", "building",
    "block", "floor", "near", "opposite", "opp", "main",
}

WEBSITE_SUFFIXES = {
    "com", "net", "org", "co", "in", "biz", "info", "us",
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


def norm_country(value) -> str:
    return safe(value).lower().strip()


def norm_name(value) -> str:
    return normalize_basic(safe(value))


def norm_no_suffix(value) -> str:
    return remove_legal_suffix(safe(value))


def compact(value) -> str:
    return normalize_compact(safe(value))


def name_tokens(value) -> list[str]:
    text = norm_name(value)
    if not text:
        return []
    return sorted({
        x for x in text.split()
        if len(x) >= 3 and x not in GENERIC_NAME
    })


def translit_tokens(value) -> list[str]:
    text = transliterate(safe(value))
    if not text:
        return []
    return sorted({
        x for x in text.split()
        if len(x) >= 3 and x not in GENERIC_NAME
    })


def address_tokens(value) -> list[str]:
    text = norm_name(value)
    if not text:
        return []
    return sorted({
        x for x in text.split()
        if len(x) >= 4 and x not in GENERIC_ADDRESS
    })


def address_pairs(value) -> list[tuple[str, str]]:
    tokens = address_tokens(value)
    out = []
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            out.append((tokens[i], tokens[j]))
    return out


def first_number(value) -> str:
    nums = extract_numbers(safe(value))
    if not nums:
        return ""
    for n in nums:
        if len(n) <= 5:
            return n
    return nums[0]


def postal_codes(value) -> list[str]:
    return [
        n for n in extract_numbers(safe(value))
        if len(n) in (5, 6)
    ]


def domain_name(value) -> str:
    text = norm_name(value)
    if not text:
        return ""
    text = norm_no_suffix(text)
    tokens = text.split()
    while tokens and tokens[-1] in WEBSITE_SUFFIXES:
        tokens.pop()
    return "".join(tokens)


def pair_key(country, a, b) -> str:
    if a > b:
        a, b = b, a
    return f"{country}|{a}|{b}"


# ============================================================
# QUERY SIGNATURE
# ============================================================

def signature(row) -> dict:
    name = safe(row.business_name)
    address = safe(row.business_address)
    c = norm_country(row.country)

    basic = norm_name(name)
    no_suffix = norm_no_suffix(name)
    translit_name = transliterate(name)

    nt = name_tokens(name)
    tt = translit_tokens(name)
    at = address_tokens(address)
    ap = address_pairs(address)

    number = first_number(address)
    postal = postal_codes(address)

    return {
        "entity_id": safe(row.entity_id),
        "country": c,
        "name": name,
        "address": address,
        "basic_name": basic,
        "no_suffix": no_suffix,
        "translit_name": translit_name,
        "domain_name": domain_name(name),
        "name_tokens": nt,
        "translit_tokens": tt,
        "address_tokens": at,
        "address_pairs": ap,
        "number": number,
        "postal": postal,
    }


# ============================================================
# STRONG KEY GENERATION
# ============================================================

def strong_keys(sig: dict) -> dict[str, list[str]]:
    c = sig["country"]
    out = defaultdict(list)

    if sig["basic_name"]:
        out["exact_name"].append(
            f"{c}|{sig['basic_name']}"
        )

    if sig["no_suffix"]:
        out["name_no_suffix"].append(
            f"{c}|{sig['no_suffix']}"
        )

    if sig["translit_name"]:
        out["translit_name"].append(
            f"{c}|{sig['translit_name']}"
        )

    if sig["domain_name"]:
        out["domain_name"].append(
            f"{c}|{sig['domain_name']}"
        )

    for p in sig["postal"]:
        for n in sig["name_tokens"]:
            out["name_postal"].append(
                f"{c}|{n}|{p}"
            )

    if sig["number"]:
        for n in sig["name_tokens"]:
            out["name_number"].append(
                f"{c}|{n}|{sig['number']}"
            )

    for n in sig["name_tokens"]:
        for a in sig["address_tokens"]:
            out["name_address"].append(
                f"{c}|{n}|{a}"
            )

    nt = sig["name_tokens"]
    for i in range(len(nt)):
        for j in range(i + 1, len(nt)):
            out["name_pair"].append(
                pair_key(c, nt[i], nt[j])
            )

    return out


# ============================================================
# RESCUE KEY GENERATION
# ============================================================

def rescue_values(sig: dict) -> dict[str, list[str]]:
    c = sig["country"]
    out = defaultdict(list)

    for t in sig["name_tokens"]:
        out["name_token"].append(f"{c}|{t}")

    for t in sig["translit_tokens"]:
        out["translit_token"].append(f"{c}|{t}")

    for t in sig["address_tokens"]:
        out["address_token"].append(f"{c}|{t}")

    for a, b in sig["address_pairs"]:
        out["address_pair"].append(
            pair_key(c, a, b)
        )

    if sig["number"]:
        for a in sig["address_tokens"]:
            out["address_number"].append(
                f"{c}|{sig['number']}|{a}"
            )

    return out


# ============================================================
# VALIDATION DATA
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

    signatures = {}
    for row in s1.itertuples(index=False):
        sig = signature(row)
        signatures[sig["entity_id"]] = sig

    truth = {}
    for row in gt.itertuples(index=False):
        value = safe(row.matched_entity_ids)
        truth[row.source1_entity_id] = (
            {x.strip() for x in value.split(",") if x.strip()}
            if value else set()
        )

    return signatures, truth


# ============================================================
# INDEX CONTAINERS
# ============================================================

@dataclass
class SourceIndex:
    strong: dict[str, dict[str, list[str]]]
    rescue: dict[str, dict[str, list[str]]]
    rescue_df: dict[str, dict[str, int]]


# ============================================================
# QUERY UNIVERSE
# ============================================================

def build_query_universe(signatures):
    strong_universe = {
        f: set() for f in STRONG_LIMITS
    }
    rescue_universe = {
        "name_token": set(),
        "translit_token": set(),
        "address_token": set(),
        "address_pair": set(),
        "address_number": set(),
    }

    for sig in signatures.values():
        for family, keys in strong_keys(sig).items():
            strong_universe[family].update(keys)

        for family, keys in rescue_values(sig).items():
            rescue_universe[family].update(keys)

    return strong_universe, rescue_universe


# ============================================================
# PASS 1: COUNT SOURCE KEY FREQUENCIES
# ============================================================

def count_source_frequencies(
    filename: str,
    source_name: str,
    strong_universe,
    rescue_universe,
):
    print()
    print("=" * 70)
    print(f"PASS 1 — COUNTING {source_name} BLOCK FREQUENCIES")
    print("=" * 70)

    counters = {
        family: Counter()
        for family in list(strong_universe) + list(rescue_universe)
    }

    rows_seen = 0
    start = time.time()

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
            f"{source_name}: {rows_seen:,} rows scanned",
            end="\r",
        )

        for row in chunk.itertuples(index=False):
            sig = signature(row)

            # Count each key at most once for this record.
            for family, keys in strong_keys(sig).items():
                universe = strong_universe[family]
                counter = counters[family]

                for key in set(keys):
                    if key in universe:
                        counter[key] += 1

            for family, keys in rescue_values(sig).items():
                universe = rescue_universe[family]
                counter = counters[family]

                for key in set(keys):
                    if key in universe:
                        # Stop tracking huge rescue keys after the
                        # threshold; they can never become useful.
                        if counter.get(key, 0) <= MAX_RESCUE_DF:
                            counter[key] += 1

    print()
    print(
        f"Finished {source_name} frequency pass in "
        f"{time.time() - start:.1f}s"
    )

    return counters


# ============================================================
# SELECT RAREST RESCUE KEYS PER QUERY
# ============================================================

def select_rescue_keys(
    signatures,
    counters,
):
    selected = {}

    family_to_source_field = {
        "name_token": "name_tokens",
        "translit_token": "translit_tokens",
        "address_token": "address_tokens",
        "address_pair": "address_pairs",
    }

    budgets = {
        "name_token": RAREST_NAME_TOKENS,
        "translit_token": RAREST_TRANSLIT_TOKENS,
        "address_token": RAREST_ADDRESS_TOKENS,
        "address_pair": RAREST_ADDRESS_PAIRS,
    }

    for sid, sig in signatures.items():
        selected[sid] = {
            "name_token": [],
            "translit_token": [],
            "address_token": [],
            "address_pair": [],
            "address_number": [],
        }

        c = sig["country"]

        for family, field in family_to_source_field.items():
            scored = []

            for value in sig[field]:
                if family == "address_pair":
                    key = pair_key(c, value[0], value[1])
                else:
                    key = f"{c}|{value}"

                df = counters[family].get(key, 0)

                if 0 < df <= MAX_RESCUE_DF:
                    scored.append((df, key))

            scored.sort(key=lambda x: x[0])

            selected[sid][family] = [
                key for _, key in scored[:budgets[family]]
            ]

        # Address number + token: choose the rarest one.
        if sig["number"] and sig["address_tokens"]:
            scored = []

            for a in sig["address_tokens"]:
                key = (
                    f"{c}|{sig['number']}|{a}"
                )
                df = counters["address_number"].get(key, 0)
                if 0 < df <= MAX_RESCUE_DF:
                    scored.append((df, key))

            scored.sort(key=lambda x: x[0])
            selected[sid]["address_number"] = [
                key
                for _, key in scored[:RAREST_ADDRESS_NUMBERS]
            ]

    return selected


# ============================================================
# PASS 2: BUILD POSTINGS FOR USABLE KEYS
# ============================================================

def build_source_index(
    filename: str,
    source_name: str,
    signatures,
    strong_universe,
    rescue_universe,
    counters,
    selected_rescue,
):
    print()
    print("=" * 70)
    print(f"PASS 2 — BUILDING {source_name} POSTINGS")
    print("=" * 70)

    strong_postings = {
        family: defaultdict(list)
        for family in STRONG_LIMITS
    }

    rescue_postings = {
        family: defaultdict(list)
        for family in rescue_universe
    }

    # Global set of rescue keys selected by at least one S1.
    selected_global = {
        family: set()
        for family in rescue_universe
    }

    for sid, family_map in selected_rescue.items():
        for family, keys in family_map.items():
            selected_global[family].update(keys)

    rows_seen = 0
    start = time.time()

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
            f"{source_name}: {rows_seen:,} rows scanned",
            end="\r",
        )

        for row in chunk.itertuples(index=False):
            entity_id = safe(row.entity_id)
            sig = signature(row)

            # ----------------------------
            # Strong blocks
            # ----------------------------
            for family, keys in strong_keys(sig).items():
                universe = strong_universe[family]
                postings = strong_postings[family]
                limit = STRONG_LIMITS[family]

                for key in set(keys):
                    if key not in universe:
                        continue

                    current = postings.get(key)

                    if current is None:
                        continue

                    if len(current) >= limit:
                        postings[key] = None
                    else:
                        current.append(entity_id)

            # ----------------------------
            # Rescue blocks
            # ----------------------------
            for family, keys in rescue_values(sig).items():
                if not keys:
                    continue

                postings = rescue_postings[family]
                selected_family = selected_global[family]

                for key in set(keys):
                    if key not in selected_family:
                        continue

                    # Frequency was already measured. This key is
                    # guaranteed to be <= MAX_RESCUE_DF.
                    current = postings.get(key)

                    if current is None:
                        current = []
                        postings[key] = current

                    current.append(entity_id)

    print()
    print(
        f"Finished {source_name} posting pass in "
        f"{time.time() - start:.1f}s"
    )

    # Convert capped-too-large strong blocks to absent blocks.
    for family, posting_map in strong_postings.items():
        strong_postings[family] = {
            k: v
            for k, v in posting_map.items()
            if v is not None
        }

    # Statistics.
    print("Strong posting keys:")
    for family in STRONG_LIMITS:
        print(
            f"  {family:20s}: "
            f"{len(strong_postings[family]):,}"
        )

    print("Rescue posting keys:")
    for family in rescue_universe:
        print(
            f"  {family:20s}: "
            f"{len(rescue_postings[family]):,}"
        )

    return SourceIndex(
        strong=strong_postings,
        rescue=dict(rescue_postings),
        rescue_df={
            family: dict(counter)
            for family, counter in counters.items()
        },
    )


# ============================================================
# STRONG CANDIDATES
# ============================================================

def get_strong_candidates(
    sig,
    index: SourceIndex,
):
    candidates = set()
    evidence = set()

    for family, keys in strong_keys(sig).items():
        for key in set(keys):
            values = index.strong[family].get(key, [])
            if values:
                candidates.update(values)
                if family in {
                    "exact_name",
                    "name_no_suffix",
                    "translit_name",
                    "domain_name",
                }:
                    evidence.add(family)

    return candidates, evidence


# ============================================================
# RESCUE CANDIDATES
# ============================================================

def get_rescue_candidates(
    sid,
    selected,
    index: SourceIndex,
    max_df: int,
):
    candidates = set()

    for family, keys in selected.get(sid, {}).items():
        postings = index.rescue.get(family, {})
        dfs = index.rescue_df.get(family, {})

        for key in keys:
            if dfs.get(key, 0) <= max_df:
                candidates.update(
                    postings.get(key, [])
                )

    return candidates


# ============================================================
# EVALUATION
# ============================================================

def evaluate_configuration(
    signatures,
    truth,
    index_s2,
    index_s3,
    selected_s2,
    selected_s3,
    rescue_df_threshold: int,
):
    total_true = 0
    total_found = 0
    candidate_counts = []
    missing = []

    for sid, true_ids in truth.items():
        sig = signatures[sid]

        s2_strong, s2_evidence = get_strong_candidates(
            sig, index_s2
        )
        s3_strong, s3_evidence = get_strong_candidates(
            sig, index_s3
        )

        strong_union = s2_strong | s3_strong

        # Only hard entities get rescue candidates.
        if len(strong_union) < HARD_CANDIDATE_THRESHOLD:
            s2_rescue = get_rescue_candidates(
                sid,
                selected_s2,
                index_s2,
                rescue_df_threshold,
            )
            s3_rescue = get_rescue_candidates(
                sid,
                selected_s3,
                index_s3,
                rescue_df_threshold,
            )
        else:
            s2_rescue = set()
            s3_rescue = set()

        candidates = (
            s2_strong
            | s3_strong
            | s2_rescue
            | s3_rescue
        )

        found = true_ids & candidates

        total_true += len(true_ids)
        total_found += len(found)
        candidate_counts.append(len(candidates))

        missing.extend(
            (sid, x)
            for x in (true_ids - candidates)
        )

    recall = (
        total_found / total_true
        if total_true
        else 0.0
    )

    counts = pd.Series(candidate_counts)

    return {
        "recall": recall,
        "total_true": total_true,
        "total_found": total_found,
        "avg": counts.mean(),
        "median": counts.median(),
        "p95": counts.quantile(.95),
        "max": counts.max(),
        "missing": missing,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    overall_start = time.time()

    signatures, truth = load_validation()

    print(
        f"\nValidation S1: {len(signatures):,}"
    )
    print(
        f"Known true matches: "
        f"{sum(len(x) for x in truth.values()):,}"
    )

    # --------------------------------------------------------
    # Build query universe
    # --------------------------------------------------------

    print("\nBuilding query universe...")

    strong_universe, rescue_universe = (
        build_query_universe(signatures)
    )

    print("Strong query keys:")
    for family, keys in strong_universe.items():
        print(
            f"  {family:20s}: {len(keys):,}"
        )

    print("Rescue query keys:")
    for family, keys in rescue_universe.items():
        print(
            f"  {family:20s}: {len(keys):,}"
        )

    # --------------------------------------------------------
    # Pass 1: frequencies
    # --------------------------------------------------------

    freq_s2 = count_source_frequencies(
        S2_FILE,
        "SOURCE2",
        strong_universe,
        rescue_universe,
    )

    freq_s3 = count_source_frequencies(
        S3_FILE,
        "SOURCE3",
        strong_universe,
        rescue_universe,
    )

    # --------------------------------------------------------
    # Select rare rescue keys per S1
    # --------------------------------------------------------

    selected_s2 = select_rescue_keys(
        signatures,
        freq_s2,
    )

    selected_s3 = select_rescue_keys(
        signatures,
        freq_s3,
    )

    print()
    print("Selected rescue keys:")

    for family in rescue_universe:
        s2_count = sum(
            len(v[family])
            for v in selected_s2.values()
        )
        s3_count = sum(
            len(v[family])
            for v in selected_s3.values()
        )

        print(
            f"  {family:20s} "
            f"S2={s2_count:,} "
            f"S3={s3_count:,}"
        )

    # --------------------------------------------------------
    # Pass 2: postings
    # --------------------------------------------------------

    index_s2 = build_source_index(
        S2_FILE,
        "SOURCE2",
        signatures,
        strong_universe,
        rescue_universe,
        freq_s2,
        selected_s2,
    )

    index_s3 = build_source_index(
        S3_FILE,
        "SOURCE3",
        signatures,
        strong_universe,
        rescue_universe,
        freq_s3,
        selected_s3,
    )

    # --------------------------------------------------------
    # Compare rescue thresholds without another source scan.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ADAPTIVE HYBRID BLOCKING RESULTS")
    print("=" * 70)

    results = []

    for threshold in [0] + RESCUE_DF_THRESHOLDS:
        result = evaluate_configuration(
            signatures,
            truth,
            index_s2,
            index_s3,
            selected_s2,
            selected_s3,
            threshold,
        )

        results.append((threshold, result))

        label = (
            "BASE ONLY"
            if threshold == 0
            else f"RESCUE DF <= {threshold}"
        )

        print()
        print(label)
        print("-" * 70)
        print(
            f"Blocking recall:       "
            f"{result['recall']:.4%}"
        )
        print(
            f"Average candidates/S1: "
            f"{result['avg']:.2f}"
        )
        print(
            f"Median candidates/S1:  "
            f"{result['median']:.2f}"
        )
        print(
            f"95th percentile:       "
            f"{result['p95']:.2f}"
        )
        print(
            f"Maximum candidates:    "
            f"{result['max']:.0f}"
        )
        print(
            f"Recovered / true:      "
            f"{result['total_found']:,} / "
            f"{result['total_true']:,}"
        )

    # --------------------------------------------------------
    # Best result by recall under 1,000 avg candidates.
    # --------------------------------------------------------

    eligible = [
        (threshold, result)
        for threshold, result in results
        if result["avg"] <= 1000
    ]

    if eligible:
        best_threshold, best = max(
            eligible,
            key=lambda x: x[1]["recall"]
        )

        print()
        print("=" * 70)
        print("RECOMMENDED V5.1 CONFIGURATION")
        print("=" * 70)
        print(
            f"Rescue DF threshold: {best_threshold}"
        )
        print(
            f"Blocking recall:     {best['recall']:.4%}"
        )
        print(
            f"Average candidates:  {best['avg']:.2f}"
        )

    # --------------------------------------------------------
    # Show missing examples for the best configuration.
    # --------------------------------------------------------

    if eligible:
        _, best = max(
            eligible,
            key=lambda x: x[1]["recall"]
        )

        if best["missing"]:
            print()
            print("Missing true matches (first 30):")
            for sid, mid in best["missing"][:30]:
                print(f"{sid} -> {mid}")
        else:
            print("\nNo known true matches missed.")

    print()
    print(
        f"Total runtime: "
        f"{time.time() - overall_start:.1f}s"
    )


if __name__ == "__main__":
    main()
