from __future__ import annotations

import pandas as pd

from .blocking import build_index, generate_candidates_for_record

S1_FILE = "diagnostics/sample_s1.tsv"
S2_FILE = "diagnostics/sample_s2.tsv"
S3_FILE = "diagnostics/sample_s3.tsv"
GT_FILE = "diagnostics/sample_ground_truth.tsv"


def load_data():
    print("Loading sampled Amazon data...")

    s1 = pd.read_csv(
        S1_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s2 = pd.read_csv(
        S2_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    s3 = pd.read_csv(
        S3_FILE,
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

    print(f"S1 records: {len(s1):,}")
    print(f"S2 records: {len(s2):,}")
    print(f"S3 records: {len(s3):,}")
    print(f"Ground truth rows: {len(gt):,}")

    return s1, s2, s3, gt


def build_truth(gt):
    truth = {}

    for _, row in gt.iterrows():

        s1_id = row["source1_entity_id"]

        raw_ids = row["matched_entity_ids"]

        if not raw_ids:
            truth[s1_id] = set()
        else:
            truth[s1_id] = {
                x.strip()
                for x in raw_ids.split(",")
                if x.strip()
            }

    return truth


def main():

    s1, s2, s3, gt = load_data()

    print("\nBuilding S2 index...")
    index_s2 = build_index(s2)

    print("Building S3 index...")
    index_s3 = build_index(s3)

    truth = build_truth(gt)

    total_true = 0
    total_found = 0

    candidate_counts = []

    missing_examples = []

    print("\nTesting blocking...\n")

    for _, row in s1.iterrows():

        s1_id = row["entity_id"]

        true_ids = truth.get(
            s1_id,
            set(),
        )

        # Generate S2 candidates
        s2_candidates = generate_candidates_for_record(
            row,
            index_s2,
        )

        # Generate S3 candidates
        s3_candidates = generate_candidates_for_record(
            row,
            index_s3,
        )

        candidates = (
            s2_candidates |
            s3_candidates
        )

        candidate_counts.append(len(candidates))

        found = true_ids & candidates

        total_true += len(true_ids)
        total_found += len(found)

        missing = true_ids - candidates

        if missing and len(missing_examples) < 20:

            for missing_id in missing:

                missing_examples.append({
                    "s1": s1_id,
                    "missing": missing_id,
                })

    print("=" * 60)
    print("BLOCKING RESULTS")
    print("=" * 60)

    if total_true > 0:
        recall = total_found / total_true
    else:
        recall = 0.0

    print(f"True matches:       {total_true:,}")
    print(f"Found by blocking:  {total_found:,}")
    print(f"Blocking recall:    {recall:.4%}")

    if candidate_counts:

        series = pd.Series(candidate_counts)

        print(f"\nAverage candidates/S1: {series.mean():.2f}")
        print(f"Median candidates/S1:  {series.median():.2f}")
        print(f"95th percentile:       {series.quantile(.95):.2f}")
        print(f"Maximum candidates:    {series.max():,}")

    if missing_examples:

        print("\nMissing true matches:")
        print(
            pd.DataFrame(
                missing_examples
            ).to_string(index=False)
        )

    else:
        print("\nNo true matches were missed.")


if __name__ == "__main__":
    main()