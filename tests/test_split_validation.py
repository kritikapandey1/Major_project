
import pytest

from src.data.splits import validate_group_splits


def test_valid_splits():
    """Accept partitions with no shared groups."""
    validate_group_splits(
        train_groups=[1, 2, 3],
        val_groups=[4, 5],
        test_groups=[6, 7],
    )


def test_train_validation_overlap():
    """Reject groups shared by train and validation."""
    with pytest.raises(ValueError, match="train and validation"):
        validate_group_splits([1, 2], [2, 3], [4, 5])


def test_train_test_overlap():
    """Reject groups shared by train and test."""
    with pytest.raises(ValueError, match="train and test"):
        validate_group_splits([1, 2], [3, 4], [2, 5])


def test_validation_test_overlap():
    """Reject groups shared by validation and test."""
    with pytest.raises(ValueError, match="validation and test"):
        validate_group_splits([1, 2], [3, 4], [4, 5])


def test_empty_splits():
    """Accept empty partitions."""
    validate_group_splits([], [], [])
