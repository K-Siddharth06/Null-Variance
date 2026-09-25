from __future__ import annotations

from collections import Counter
import pandas as pd
import re

from .normalize import (
    normalize_basic,
    remove_legal_suffix,
    extract_numbers,
)


BASE = (
    r"C:\Users\darsh\Downloads"
    r"\6ab10eb3b23ba_student_resource"
    r"\student_resource\dataset\train"
)

S1_FILE = BASE + r"\train_source1.tsv"
S2_FILE = BASE + r"\train_source2.tsv"
S3_FILE = BASE + r"\train_source3.tsv"

SAMPLE_S1 = r"diagnostics\sample_s1.tsv"

CHUNK_SIZE = 250_000


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


def safe(value) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def first_token(name: str) -> str:

    name = normalize_basic(name)

    if not name:
        return ""

    for token in name.split():

        if len(token) < 3:
            continue

        if token in GENERIC_NAME_TOKENS:
            continue

        return token

    return ""


def name_tokens(name: str) -> list[str]:

    name = normalize_basic(name)

    if not name:
        return []

    result = []

    for token in name.split():

        if len(token) < 3:
            continue

        if token in GENERIC_NAME_TOKENS:
            continue

        result.append(token)

    return result


def address_tokens(address: str) -> list[str]:

    generic = {
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

    address = normalize_basic(address)

    if not address:
        return []

    return [
        x
        for x in address.split()
        if len(x) >= 4 and x not in generic
    ]


def first_number(address: str) -> str:

    numbers = extract_numbers(address)

    if not numbers:
        return ""

    for number in numbers:

        if len(number) <= 5:
            return number

    return numbers[0]


def postal_codes(address: str) -> list[str]:

    return [
        x
        for x in extract_numbers(address)
        if len(x) in (5, 6)
    ]


def build_query_keys():

    s1 = pd.read_csv(
        SAMPLE_S1,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    maps = {
        "exact_name": set(),
        "name_no_suffix": set(),
        "prefix5": set(),
        "first_token": set(),
        "name_token": set(),
        "address_number": set(),
        "postal": set(),
        "address_token": set(),
    }

    for row in s1.itertuples(index=False):

        country = safe(row.country).lower()

        name = safe(row.business_name)
        address = safe(row.business_address)

        basic = normalize_basic(name)
        no_suffix = remove_legal_suffix(name)

        if basic:
            maps["exact_name"].add(
                f"{country}|{basic}"
            )

        if no_suffix:
            maps["name_no_suffix"].add(
                f"{country}|{no_suffix}"
            )

        compact = re.sub(
            r"\s+",
            "",
            basic,
        )

        if compact:
            maps["prefix5"].add(
                f"{country}|{compact[:5]}"
            )

        token = first_token(name)

        if token:
            maps["first_token"].add(
                f"{country}|{token}"
            )

        for token in name_tokens(name):

            maps["name_token"].add(
                f"{country}|{token}"
            )

        number = first_number(address)

        if number:
            maps["address_number"].add(
                f"{country}|{number}"
            )

        for postal in postal_codes(address):

            maps["postal"].add(
                f"{country}|{postal}"
            )

        for token in address_tokens(address):

            maps["address_token"].add(
                f"{country}|{token}"
            )

    return maps


def profile_source(
    filename: str,
    source_name: str,
    query_keys,
):

    print()
    print("=" * 70)
    print(f"PROFILE: {source_name}")
    print("=" * 70)

    counters = {
        key: Counter()
        for key in query_keys
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

        countries = (
            chunk["country"]
            .astype(str)
            .str.lower()
            .str.strip()
            .tolist()
        )

        names = chunk["business_name"].tolist()
        addresses = chunk["business_address"].tolist()

        for i in range(len(chunk)):

            country = countries[i]

            name = safe(names[i])
            address = safe(addresses[i])

            basic = normalize_basic(name)
            no_suffix = remove_legal_suffix(name)

            # -----------------------------
            # Exact name
            # -----------------------------

            if basic:

                key = f"{country}|{basic}"

                if key in query_keys["exact_name"]:
                    counters["exact_name"][key] += 1

            # -----------------------------
            # Name without suffix
            # -----------------------------

            if no_suffix:

                key = f"{country}|{no_suffix}"

                if key in query_keys["name_no_suffix"]:
                    counters["name_no_suffix"][key] += 1

            # -----------------------------
            # Prefix
            # -----------------------------

            compact = re.sub(
                r"\s+",
                "",
                basic,
            )

            if compact:

                key = (
                    f"{country}|"
                    f"{compact[:5]}"
                )

                if key in query_keys["prefix5"]:
                    counters["prefix5"][key] += 1

            # -----------------------------
            # First token
            # -----------------------------

            token = first_token(name)

            if token:

                key = f"{country}|{token}"

                if key in query_keys["first_token"]:
                    counters["first_token"][key] += 1

            # -----------------------------
            # Name tokens
            # -----------------------------

            for token in name_tokens(name):

                key = f"{country}|{token}"

                if key in query_keys["name_token"]:
                    counters["name_token"][key] += 1

            # -----------------------------
            # Address number
            # -----------------------------

            number = first_number(address)

            if number:

                key = f"{country}|{number}"

                if key in query_keys["address_number"]:
                    counters["address_number"][key] += 1

            # -----------------------------
            # Postal
            # -----------------------------

            for postal in postal_codes(address):

                key = f"{country}|{postal}"

                if key in query_keys["postal"]:
                    counters["postal"][key] += 1

            # -----------------------------
            # Address token
            # -----------------------------

            for token in address_tokens(address):

                key = f"{country}|{token}"

                if key in query_keys["address_token"]:
                    counters["address_token"][key] += 1

    print()

    # ========================================================
    # Statistics
    # ========================================================

    results = []

    for family, counter in counters.items():

        values = list(counter.values())

        if not values:

            print(
                f"{family:20s} "
                f"NO MATCHING QUERY KEYS"
            )

            continue

        s = pd.Series(values)

        result = {
            "family": family,
            "keys_seen": len(values),
            "total_rows": sum(values),
            "median": s.median(),
            "p90": s.quantile(0.90),
            "p95": s.quantile(0.95),
            "p99": s.quantile(0.99),
            "max": s.max(),
            "le_10": (s <= 10).mean() * 100,
            "le_25": (s <= 25).mean() * 100,
            "le_50": (s <= 50).mean() * 100,
            "le_100": (s <= 100).mean() * 100,
            "le_500": (s <= 500).mean() * 100,
        }

        results.append(result)

    if results:

        result_df = pd.DataFrame(results)

        print(
            result_df.to_string(
                index=False
            )
        )

        output_file = (
            f"diagnostics/"
            f"{source_name.lower()}_block_profile.csv"
        )

        result_df.to_csv(
            output_file,
            index=False,
        )

        print(
            f"\nSaved: {output_file}"
        )

    # ========================================================
    # Show worst keys
    # ========================================================

    for family, counter in counters.items():

        if not counter:
            continue

        print()
        print(
            f"Top 10 largest {family} blocks:"
        )

        for key, count in counter.most_common(10):

            print(
                f"{count:8d}  {key}"
            )


def main():

    print("Building query keys...")

    query_keys = build_query_keys()

    for family, keys in query_keys.items():

        print(
            f"{family:20s}: "
            f"{len(keys):,} query keys"
        )

    profile_source(
        S2_FILE,
        "SOURCE2",
        query_keys,
    )

    profile_source(
        S3_FILE,
        "SOURCE3",
        query_keys,
    )


if __name__ == "__main__":
    main()