import numpy as np
import pytest

from src.group_cv import (
    InsufficientGroupsError,
    OuterSplit,
    audit_splits,
    inner_group_split,
    make_repeated_splits,
    randomized_group_kfold,
)

SEEDS = [11, 23, 37, 41, 53]


def _groups(sizes):
    return np.concatenate([[f"g{i}"] * n for i, n in enumerate(sizes)])


def test_every_group_is_tested_exactly_once_per_repeat_and_never_split():
    groups = _groups([11, 7, 7, 6, 6, 5, 4, 3, 3, 2])  # same shape as the MaCFP laboratories
    for seed in SEEDS:
        folds = randomized_group_kfold(groups, 5, seed)
        tested = np.concatenate(folds)
        assert sorted(tested) == list(range(len(groups)))  # each row tested once
        for fold in folds:
            for g in set(groups[fold]):
                assert set(np.flatnonzero(groups == g)) <= set(fold)  # whole group in one fold


def test_partitions_are_group_disjoint_and_audit_passes():
    groups = _groups([11, 7, 7, 6, 6, 5, 4, 3, 3, 2])
    ids = np.array([f"r{i}" for i in range(len(groups))])
    splits = make_repeated_splits(groups, SEEDS, n_splits=5)
    assert len(splits) == 25
    for s in splits:
        tr, va, te = set(groups[s.inner_train_idx]), set(groups[s.inner_val_idx]), set(groups[s.test_idx])
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert len(va) >= 1 and len(tr) >= 2
    audit = audit_splits(splits, groups, ids, n_splits=5)
    assert audit.passed, audit.failures
    assert all(v == 0 for v in audit.checks.values())


def test_splits_are_reproducible_and_seeds_differ():
    groups = _groups([5] * 12)
    a = make_repeated_splits(groups, SEEDS)
    b = make_repeated_splits(groups, SEEDS)
    assert all(np.array_equal(x.test_idx, y.test_idx) and np.array_equal(x.inner_val_idx, y.inner_val_idx)
               for x, y in zip(a, b))
    first_folds = {seed: tuple(sorted(set(groups[s.test_idx]))) for s in a if s.fold == 1 for seed in [s.seed]}
    assert len(set(first_folds.values())) > 1, "different seeds must give different fold assignments"


def test_too_few_groups_is_refused_not_silently_evaluated():
    with pytest.raises(InsufficientGroupsError):
        randomized_group_kfold(_groups([3, 3, 3, 3]), 5, seed=1)
    with pytest.raises(InsufficientGroupsError):
        inner_group_split(_groups([3, 3]), np.arange(6), seed=1)


def test_audit_detects_a_leaky_split():
    groups = _groups([4, 4, 4, 4, 4, 4])
    ids = np.array([f"r{i}" for i in range(len(groups))])
    good = make_repeated_splits(groups, [1], n_splits=5)
    leaky_test = np.concatenate([good[0].test_idx, good[0].inner_train_idx[:1]])  # one training row also in test
    bad = [OuterSplit(1, 1, good[0].inner_train_idx, good[0].inner_val_idx, np.sort(leaky_test)), *good[1:]]
    audit = audit_splits(bad, groups, ids, n_splits=5)
    assert not audit.passed
    assert audit.checks["train_test_group_overlaps"] >= 1
    assert audit.checks["record_ids_in_conflicting_partitions"] >= 1
