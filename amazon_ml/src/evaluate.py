from __future__ import annotations

from typing import Iterable, Set


def f05_score(
    true_ids: Iterable[str],
    predicted_ids: Iterable[str],
) -> float:
    """
    Calculate F0.5 for one Source-1 entity.

    F0.5 weights precision more heavily than recall.
    """

    true_set: Set[str] = set(true_ids)
    pred_set: Set[str] = set(predicted_ids)

    # Singleton with correct empty prediction.
    if not true_set and not pred_set:
        return 1.0

    # Singleton with false prediction.
    if not true_set and pred_set:
        return 0.0

    # No prediction when matches exist.
    if true_set and not pred_set:
        return 0.0

    tp = len(true_set & pred_set)

    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    if precision == 0 or recall == 0:
        return 0.0

    beta = 0.5

    return (
        (1 + beta**2) * precision * recall
        / ((beta**2 * precision) + recall)
    )


def macro_f05(
    truth: dict[str, Set[str]],
    predictions: dict[str, Set[str]],
) -> float:
    """
    Calculate macro-average F0.5 across Source-1 entities.
    """

    scores = []

    for s1_id, true_ids in truth.items():

        predicted_ids = predictions.get(s1_id, set())

        scores.append(
            f05_score(
                true_ids,
                predicted_ids,
            )
        )

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def blocking_recall(
    truth: dict[str, Set[str]],
    candidates: dict[str, Set[str]],
) -> float:
    """
    Measure candidate-generation recall.

    Of all true matches, how many survived blocking?
    """

    total_true = 0
    total_found = 0

    for s1_id, true_ids in truth.items():

        candidate_ids = candidates.get(
            s1_id,
            set(),
        )

        total_true += len(true_ids)

        total_found += len(
            true_ids & candidate_ids
        )

    if total_true == 0:
        return 0.0

    return total_found / total_true


if __name__ == "__main__":

    # Small sanity test

    truth = {
        "S1-001": {"S2-001", "S3-001"},
        "S1-002": set(),
        "S1-003": {"S2-010"},
    }

    predictions = {
        "S1-001": {"S2-001", "S3-001"},
        "S1-002": set(),
        "S1-003": {"S2-999"},
    }

    candidates = {
        "S1-001": {"S2-001", "S3-001", "S2-999"},
        "S1-002": set(),
        "S1-003": {"S2-010", "S2-999"},
    }

    print(
        "F0.5:",
        macro_f05(truth, predictions),
    )

    print(
        "Blocking recall:",
        blocking_recall(truth, candidates),
    )