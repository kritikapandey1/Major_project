
"""Utilities for validating dataset group splits."""


def validate_group_splits(train_groups, val_groups, test_groups):
    """Ensure no group appears in more than one data partition.

    Args:
        train_groups: Group IDs assigned to the training partition.
        val_groups: Group IDs assigned to the validation partition.
        test_groups: Group IDs assigned to the test partition.

    Raises:
        ValueError: If any group appears in multiple partitions.
    """
    train = set(train_groups)
    validation = set(val_groups)
    test = set(test_groups)

    if train & validation:
        raise ValueError("Overlap found between train and validation sets.")

    if train & test:
        raise ValueError("Overlap found between train and test sets.")

    if validation & test:
        raise ValueError("Overlap found between validation and test sets.")
