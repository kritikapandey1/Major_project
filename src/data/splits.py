
"""Utilities for validating dataset group splits.

These helpers check that a group is assigned to only one
partition. They do not generate the partitions themselves.
"""


def find_group_overlaps(train_groups, val_groups, test_groups):
    """Find groups shared between different data partitions.

    Args:
        train_groups: Group IDs in the training partition.
        val_groups: Group IDs in the validation partition.
        test_groups: Group IDs in the test partition.

    Returns:
        A dictionary containing only partition pairs that overlap.
        Each value is a set of the shared group IDs.
    """
    partitions = {
        "train": set(train_groups),
        "validation": set(val_groups),
        "test": set(test_groups),
    }

    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    overlaps = {}

    for first, second in pairs:
        shared_groups = partitions[first] & partitions[second]

        if shared_groups:
            pair_name = f"{first}-{second}"
            overlaps[pair_name] = shared_groups

    return overlaps


def validate_group_splits(train_groups, val_groups, test_groups):
    """Raise ValueError if groups appear in multiple partitions.

    Args:
        train_groups: Group IDs assigned to training.
        val_groups: Group IDs assigned to validation.
        test_groups: Group IDs assigned to testing.

    Raises:
        ValueError: If any group is shared between partitions.
    """
    overlaps = find_group_overlaps(
        train_groups,
        val_groups,
        test_groups,
    )

    if overlaps:
        details = []

        for pair, groups in overlaps.items():
            readable_groups = sorted(repr(group) for group in groups)
            details.append(f"{pair}: {readable_groups}")

        message = "; ".join(details)
        raise ValueError(f"Group overlap detected: {message}")
