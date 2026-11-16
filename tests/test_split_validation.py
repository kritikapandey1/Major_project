
import pytest

from src.data.splits import (
    find_group_overlaps,
    validate_group_splits,
)


def test_valid_splits():
    """Accept partitions that have no shared groups."""
    result = find_group_overlaps(
        [1, 2, 3],
        [4, 5],
        [6, 7],
    )
    assert result == {}


def test_no_error_for_valid_splits():
    """Validation should complete without raising an error."""
    validate_group_splits(
        train_groups=[1, 2, 3],
        val_groups=[4, 5],
        test_groups=[6, 7],
    )


def test_train_validation_overlap():
    """Detect groups shared by train and validation."""
    result = find_group_overlaps(
        [1, 2],
        [2, 3],
        [4, 5],
    )
    assert result["train-validation"] == {2}


def test_train_test_overlap():
    """Detect groups shared by train and test."""
    result = find_group_overlaps(
        [1, 2],
        [3, 4],
        [2, 5],
    )
    assert result["train-test"] == {2}


def test_validation_test_overlap():
    """Detect groups shared by validation and test."""
    result = find_group_overlaps(
        [1, 2],
        [3, 4],
        [4, 5],
    )
    assert result["validation-test"] == {4}


def test_multiple_overlaps_are_detected():
    """Report overlaps across all three partition pairs."""
    result = find_group_overlaps(
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
    )

    assert result == {
        "train-validation": {2, 3},
        "train-test": {3},
        "validation-test": {3, 4},
    }


def test_validator_reports_multiple_overlaps():
    """The error message should include every overlapping pair."""
    with pytest.raises(ValueError) as error:
        validate_group_splits(
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
        )

    message = str(error.value)
    assert "train-validation" in message
    assert "train-test" in message
    assert "validation-test" in message


def test_empty_splits():
    """Accept three empty partitions."""
    validate_group_splits([], [], [])
    assert find_group_overlaps([], [], []) == {}


def test_generator_inputs():
    """Accept generators as group ID inputs."""
    train = (x for x in [1, 2])
    validation = (x for x in [3, 4])
    test = (x for x in [5, 6])

    validate_group_splits(train, validation, test)


def test_repeated_groups_within_one_partition():
    """Repeated IDs within one partition do not cause overlap."""
    validate_group_splits(
        [1, 1, 2, 2],
        [3, 3, 4],
        [5, 5, 6],
    )


@pytest.mark.parametrize(
    "train, validation, test, expected",
    [
        ([1, 2], [2, 3], [4], "train-validation"),
        ([1, 2], [3, 4], [2, 5], "train-test"),
        ([1, 2], [3, 4], [4, 5], "validation-test"),
    ],
)
def test_validator_rejects_each_overlap(
    train, validation, test, expected
):
    """Reject overlap for every possible partition pair."""
    with pytest.raises(ValueError, match=expected):
        validate_group_splits(train, validation, test)
