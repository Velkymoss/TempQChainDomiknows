import math
import random
from collections import Counter


def get_train_labels(dataset: list[dict[str, str]]) -> list[int]:
    labels = []
    for batch in dataset:
        batch_labels = batch["labels"].split("@@")
        batch_labels = [int(label) for label in batch_labels]
        labels.extend(batch_labels)
    return labels


def get_class_distribution(dataset: list[dict[str, str]]) -> dict[int, dict[str, int | float]]:
    labels = get_train_labels(dataset)
    total_samples = len(labels)

    class_counts = Counter(labels)

    distribution = {}
    for class_label in sorted(class_counts.keys()):
        count = class_counts[class_label]
        distribution[class_label] = {
            "count": count,
            "percentage": (count / total_samples) * 100 if total_samples > 0 else 0.0,
        }

    return distribution


def sample_batches(batches: list[dict[str, str]], ratio: float, ratio_seed: int | None = 42) -> list[dict[str, str]]:
    """Sample a random fraction (0.0 to 1.0) of batches for training.

    Args:
        batches: List of batches.
        ratio: Fraction of batches to sample, between 0.0 and 1.0.
        ratio_seed: Optional integer seed for reproducible random selection.

    Returns:
        A list containing the sampled subset of batches.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"Ratio must be between 0.0 and 1.0 inclusive, got {ratio}")

    k = math.ceil(len(batches) * ratio)

    if ratio_seed is not None:
        # Create an isolated local generator instance
        rng = random.Random(ratio_seed)
        return rng.sample(batches, k=k)

    return random.sample(batches, k=k)
