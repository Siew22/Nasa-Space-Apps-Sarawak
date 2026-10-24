"""Seeded, group-aware repeated cross-validation splits with a built-in leakage audit.

Why a custom splitter: scikit-learn 1.5.2 (pinned in requirements.txt) ships
``GroupKFold(n_splits)`` with no ``shuffle``/``random_state`` arguments, so
"repeating" it with five seeds would produce the same five folds five times.
``randomized_group_kfold`` shuffles whole groups with a seed and then assigns
each group to the fold that currently holds the fewest rows, so every group
appears in exactly one test fold per repeat and fold sizes stay balanced.

Partitions per outer fold (all built from whole groups, never split):
    outer_test       - held out first; used once for the final score
    inner_train      - fits candidate models during model selection
    inner_validation - scores candidate models during model selection
The selected configuration is refit on outer_train = inner_train + inner_validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class InsufficientGroupsError(ValueError):
    """There are too few independent groups for the requested protocol."""


@dataclass(frozen=True)
class OuterSplit:
    seed: int
    fold: int  # 1-based
    inner_train_idx: np.ndarray
    inner_val_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def outer_train_idx(self) -> np.ndarray:
        return np.sort(np.concatenate([self.inner_train_idx, self.inner_val_idx]))


@dataclass
class SplitAudit:
    passed: bool
    checks: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def randomized_group_kfold(groups: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    """Return one array of row indices per test fold; each group lands in exactly one fold."""
    groups = np.asarray(groups)
    unique, counts = np.unique(groups, return_counts=True)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(unique) < n_splits:
        raise InsufficientGroupsError(
            f"{len(unique)} independent groups cannot form {n_splits} group-disjoint folds"
        )
    rng = np.random.default_rng(seed)
    fold_rows = np.zeros(n_splits, dtype=int)
    fold_groups: list[list] = [[] for _ in range(n_splits)]
    for g in rng.permutation(len(unique)):
        k = int(np.argmin(fold_rows))  # fewest rows so far; ties go to the lowest fold index
        fold_groups[k].append(unique[g])
        fold_rows[k] += counts[g]
    return [np.flatnonzero(np.isin(groups, members)) for members in fold_groups]


def inner_group_split(
    groups: np.ndarray, candidate_idx: np.ndarray, seed: int,
    val_fraction: float = 0.2, min_train_groups: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the outer-training rows into inner-train / inner-validation by whole groups."""
    groups = np.asarray(groups)
    candidate_idx = np.asarray(candidate_idx)
    cand_groups, cand_counts = np.unique(groups[candidate_idx], return_counts=True)
    if len(cand_groups) < min_train_groups + 1:
        raise InsufficientGroupsError(
            f"outer-training partition has {len(cand_groups)} groups; need at least {min_train_groups + 1} "
            "to hold out an inner validation group"
        )
    rng = np.random.default_rng(seed)
    target_rows = val_fraction * len(candidate_idx)
    val_groups, val_rows = [], 0
    for g in rng.permutation(len(cand_groups)):
        if len(cand_groups) - len(val_groups) <= min_train_groups:
            break
        val_groups.append(cand_groups[g])
        val_rows += cand_counts[g]
        if val_rows >= target_rows:
            break
    in_val = np.isin(groups[candidate_idx], val_groups)
    return np.sort(candidate_idx[~in_val]), np.sort(candidate_idx[in_val])


def make_repeated_splits(
    groups: np.ndarray, seeds: list[int], n_splits: int = 5, val_fraction: float = 0.2,
) -> list[OuterSplit]:
    """Five seeds x five folds = 25 outer splits, each with its own inner validation split."""
    groups = np.asarray(groups)
    all_idx = np.arange(len(groups))
    splits = []
    for seed in seeds:
        for fold, test_idx in enumerate(randomized_group_kfold(groups, n_splits, seed), start=1):
            outer_train = np.setdiff1d(all_idx, test_idx)
            inner_train, inner_val = inner_group_split(
                groups, outer_train, seed=seed * 1000 + fold, val_fraction=val_fraction
            )
            splits.append(OuterSplit(seed, fold, inner_train, inner_val, np.sort(test_idx)))
    return splits


def audit_splits(splits: list[OuterSplit], groups: np.ndarray, record_ids: np.ndarray, n_splits: int) -> SplitAudit:
    """Programmatic leakage audit; every count in ``checks`` must be zero for ``passed``."""
    groups = np.asarray(groups)
    record_ids = np.asarray(record_ids)
    checks = {
        "train_validation_group_overlaps": 0,
        "train_test_group_overlaps": 0,
        "validation_test_group_overlaps": 0,
        "record_ids_in_conflicting_partitions": 0,
        "rows_missing_from_a_split": 0,
        "groups_not_tested_exactly_once_per_repeat": 0,
    }
    failures: list[str] = []
    n = len(groups)
    for s in splits:
        tr, va, te = (set(groups[i]) for i in (s.inner_train_idx, s.inner_val_idx, s.test_idx))
        for name, a, b in (("train_validation", tr, va), ("train_test", tr, te), ("validation_test", va, te)):
            shared = a & b
            if shared:
                checks[f"{name}_group_overlaps"] += len(shared)
                failures.append(f"seed {s.seed} fold {s.fold}: {name} share groups {sorted(map(str, shared))[:5]}")
        ids = [record_ids[i] for i in (s.inner_train_idx, s.inner_val_idx, s.test_idx)]
        conflict = (set(ids[0]) & set(ids[1])) | (set(ids[0]) & set(ids[2])) | (set(ids[1]) & set(ids[2]))
        if conflict:
            checks["record_ids_in_conflicting_partitions"] += len(conflict)
            failures.append(f"seed {s.seed} fold {s.fold}: record IDs in two partitions {sorted(map(str, conflict))[:5]}")
        covered = len(s.inner_train_idx) + len(s.inner_val_idx) + len(s.test_idx)
        if covered != n or len(np.unique(np.concatenate([s.inner_train_idx, s.inner_val_idx, s.test_idx]))) != n:
            checks["rows_missing_from_a_split"] += 1
            failures.append(f"seed {s.seed} fold {s.fold}: partitions do not cover every row exactly once")
    for seed in sorted({s.seed for s in splits}):
        seed_splits = [s for s in splits if s.seed == seed]
        folds_per_group: dict = {}
        for s in seed_splits:  # count test FOLDS per group (not rows)
            for g in set(groups[s.test_idx]):
                folds_per_group[g] = folds_per_group.get(g, 0) + 1
        bad = [g for g in np.unique(groups) if folds_per_group.get(g, 0) != 1]
        if bad or len(seed_splits) != n_splits:
            checks["groups_not_tested_exactly_once_per_repeat"] += len(bad) or 1
            failures.append(f"seed {seed}: groups not tested exactly once: {sorted(map(str, bad))[:5]}")
    return SplitAudit(passed=not failures, checks=checks, failures=failures)
