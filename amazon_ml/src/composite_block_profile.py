from __future__ import annotations

import re
from collections import Counter

import pandas as pd


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


def norm(value) -> str:

    value = safe(value)

    if not value:
        return ""

    value = value.lower()

    value = value.replace("&", " and ")

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def name_tokens(value) -> list[str]:

    text = norm(value)

    if not text:
        return []

    return [
        x
        for x in set(text.split())
        if len(x) >= 3
        and x not in GENERIC_NAME
    ]


def address_tokens(value) -> list[str]:

    text = norm(value)

    if not text:
        return []

    return [
        x
        for x in set(text.split())
        if len(x) >= 4
        and x not in GENERIC_ADDRESS
    ]


def numbers(value) -> list[str]:

    return re.findall(
        r"\d+",
        safe(value),
    )


def first_number(value) -> str:

    ns = numbers(value)

    if not ns:
        return ""

    for n in ns:

        if len(n) <= 5:
            return n

    return ns[0]


def postal_codes(value) -> list[str]:

    return [
        n
        for n in numbers(value)
        if len(n) in (5, 6)
    ]


# ============================================================
# QUERY BLOCK KEYS
# ============================================================

def build_query_keys(row):

    country = norm(row["country"])

    name = safe(row["business_name"])

    address = safe(row["business_address"])

    nt = sorted(
        name_tokens(name)
    )

    at = sorted(
        address_tokens(address)
    )

    number = first_number(address)

    postal = postal_codes(address)

    result = {
        "name_pair": set(),
        "address_pair": set(),
        "name_address": set(),
        "name_number": set(),
        "name_postal": set(),
        "address_number": set(),
    }

    # --------------------------------------------------------
    # Two name tokens
    # --------------------------------------------------------

    for i in range(len(nt)):

        for j in range(i + 1, len(nt)):

            result["name_pair"].add(
                f"{country}|{nt[i]}|{nt[j]}"
            )

    # --------------------------------------------------------
    # Two address tokens
    # --------------------------------------------------------

    for i in range(len(at)):

        for j in range(i + 1, len(at)):

            result["address_pair"].add(
                f"{country}|{at[i]}|{at[j]}"
            )

    # --------------------------------------------------------
    # Name token + address token
    # --------------------------------------------------------

    for n in nt:

        for a in at:

            result["name_address"].add(
                f"{country}|{n}|{a}"
            )

    # --------------------------------------------------------
    # Name + house number
    # --------------------------------------------------------

    if number:

        for n in nt:

            result["name_number"].add(
                f"{country}|{n}|{number}"
            )

    # --------------------------------------------------------
    # Name + postal
    # --------------------------------------------------------

    for p in postal:

        for n in nt:

            result["name_postal"].add(
                f"{country}|{n}|{p}"
            )

    # --------------------------------------------------------
    # Address token + number
    # --------------------------------------------------------

    if number:

        for a in at:

            result["address_number"].add(
                f"{country}|{number}|{a}"
            )

    return result


# ============================================================
# LOAD QUERY DATA
# ============================================================

def load_queries():

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

    query_keys = {
        family: set()
        for family in [
            "name_pair",
            "address_pair",
            "name_address",
            "name_number",
            "name_postal",
            "address_number",
        ]
    }

    per_s1 = {}

    for row in s1.to_dict("records"):

        sid = row["entity_id"]

        keys = build_query_keys(
            row
        )

        per_s1[sid] = keys

        for family, values in keys.items():

            query_keys[family].update(
                values
            )

    truth = {}

    for row in gt.itertuples(
        index=False
    ):

        matched = safe(
            row.matched_entity_ids
        )

        truth[
            row.source1_entity_id
        ] = (
            {
                x.strip()
                for x in matched.split(",")
                if x.strip()
            }
            if matched
            else set()
        )

    return (
        s1,
        per_s1,
        query_keys,
        truth,
    )


# ============================================================
# PROFILE ONE SOURCE
# ============================================================

def profile_source(
    filename,
    source_name,
    query_keys,
):

    print()
    print("=" * 70)
    print(f"PROFILING {source_name}")
    print("=" * 70)

    counters = {
        family: Counter()
        for family in query_keys
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

            country = norm(
                row.country
            )

            nt = sorted(
                name_tokens(
                    row.business_name
                )
            )

            at = sorted(
                address_tokens(
                    row.business_address
                )
            )

            number = first_number(
                row.business_address
            )

            postal = postal_codes(
                row.business_address
            )

            # ------------------------------------------------
            # Name pairs
            # ------------------------------------------------

            for i in range(len(nt)):

                for j in range(i + 1, len(nt)):

                    key = (
                        f"{country}|"
                        f"{nt[i]}|{nt[j]}"
                    )

                    if key in query_keys[
                        "name_pair"
                    ]:

                        counters[
                            "name_pair"
                        ][key] += 1

            # ------------------------------------------------
            # Address pairs
            # ------------------------------------------------

            for i in range(len(at)):

                for j in range(i + 1, len(at)):

                    key = (
                        f"{country}|"
                        f"{at[i]}|{at[j]}"
                    )

                    if key in query_keys[
                        "address_pair"
                    ]:

                        counters[
                            "address_pair"
                        ][key] += 1

            # ------------------------------------------------
            # Name + address
            # ------------------------------------------------

            for n in nt:

                for a in at:

                    key = (
                        f"{country}|"
                        f"{n}|{a}"
                    )

                    if key in query_keys[
                        "name_address"
                    ]:

                        counters[
                            "name_address"
                        ][key] += 1

            # ------------------------------------------------
            # Name + number
            # ------------------------------------------------

            if number:

                for n in nt:

                    key = (
                        f"{country}|"
                        f"{n}|{number}"
                    )

                    if key in query_keys[
                        "name_number"
                    ]:

                        counters[
                            "name_number"
                        ][key] += 1

            # ------------------------------------------------
            # Name + postal
            # ------------------------------------------------

            for p in postal:

                for n in nt:

                    key = (
                        f"{country}|"
                        f"{n}|{p}"
                    )

                    if key in query_keys[
                        "name_postal"
                    ]:

                        counters[
                            "name_postal"
                        ][key] += 1

            # ------------------------------------------------
            # Address token + number
            # ------------------------------------------------

            if number:

                for a in at:

                    key = (
                        f"{country}|"
                        f"{number}|{a}"
                    )

                    if key in query_keys[
                        "address_number"
                    ]:

                        counters[
                            "address_number"
                        ][key] += 1

    print()

    return counters


# ============================================================
# SUMMARY
# ============================================================

def summarize(
    counters
):

    rows = []

    for family, counter in counters.items():

        values = list(
            counter.values()
        )

        if not values:

            continue

        s = pd.Series(values)

        rows.append({
            "family": family,
            "keys": len(values),
            "total_rows": sum(values),
            "median": s.median(),
            "p90": s.quantile(.90),
            "p95": s.quantile(.95),
            "p99": s.quantile(.99),
            "max": s.max(),

            "le_10":
                (s <= 10).mean() * 100,

            "le_25":
                (s <= 25).mean() * 100,

            "le_50":
                (s <= 50).mean() * 100,

            "le_100":
                (s <= 100).mean() * 100,

            "le_250":
                (s <= 250).mean() * 100,

            "le_500":
                (s <= 500).mean() * 100,
        })

    return pd.DataFrame(rows)


# ============================================================
# TRUE-PAIR COVERAGE
# ============================================================

def build_pair_keys_from_values(
    country,
    name,
    address,
):

    row = {
        "country": country,
        "business_name": name,
        "business_address": address,
    }

    return build_query_keys(
        row
    )


def positive_coverage(
    s1,
    truth,
    source_rows,
):

    # This only evaluates whether a TRUE positive pair
    # shares a composite key.
    #
    # It does not yet know the block size. That comes from
    # the full-source profiling above.

    records = []

    for _, row in s1.iterrows():

        sid = row["entity_id"]

        if sid not in truth:
            continue

        s1_keys = build_pair_keys_from_values(
            row["country"],
            row["business_name"],
            row["business_address"],
        )

        for target_id in truth[sid]:

            target = source_rows.get(
                target_id
            )

            if target is None:
                continue

            target_keys = (
                build_pair_keys_from_values(
                    target["country"],
                    target["business_name"],
                    target["business_address"],
                )
            )

            records.append({
                "s1": sid,
                "target": target_id,

                "name_pair":
                    bool(
                        s1_keys["name_pair"]
                        &
                        target_keys["name_pair"]
                    ),

                "address_pair":
                    bool(
                        s1_keys["address_pair"]
                        &
                        target_keys["address_pair"]
                    ),

                "name_address":
                    bool(
                        s1_keys["name_address"]
                        &
                        target_keys["name_address"]
                    ),

                "name_number":
                    bool(
                        s1_keys["name_number"]
                        &
                        target_keys["name_number"]
                    ),

                "name_postal":
                    bool(
                        s1_keys["name_postal"]
                        &
                        target_keys["name_postal"]
                    ),

                "address_number":
                    bool(
                        s1_keys["address_number"]
                        &
                        target_keys["address_number"]
                    ),
            })

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================

def main():

    start = pd.Timestamp.now()

    (
        s1,
        per_s1,
        query_keys,
        truth,
    ) = load_queries()

    print("Query block counts:")

    for family, keys in query_keys.items():

        print(
            f"{family:20s}: "
            f"{len(keys):,}"
        )

    # --------------------------------------------------------
    # Profile full sources
    # --------------------------------------------------------

    s2 = profile_source(
        S2_FILE,
        "SOURCE2",
        query_keys,
    )

    s3 = profile_source(
        S3_FILE,
        "SOURCE3",
        query_keys,
    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SOURCE 2 COMPOSITE BLOCK SUMMARY")
    print("=" * 70)

    print(
        summarize(s2).to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("SOURCE 3 COMPOSITE BLOCK SUMMARY")
    print("=" * 70)

    print(
        summarize(s3).to_string(
            index=False
        )
    )

    print()
    print(
        "Finished profiling in",
        (
            pd.Timestamp.now()
            - start
        ).total_seconds(),
        "seconds"
    )


if __name__ == "__main__":
    main()