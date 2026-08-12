"""Deterministic source-level stratified partitioning for Surv-VAU."""
from __future__ import annotations

import random
import math
from collections import defaultdict
from typing import Dict, List, Tuple


def stratified_partition(
    source_records: Dict[str, List[dict]],
    fractions: Tuple[float, ...],
    names: Tuple[str, ...],
    seed: int,
) -> Dict[str, str]:
    """Partition sources within frozen ``(source_dataset, event_type)`` strata."""
    if len(fractions) != len(names) or abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("Partition fractions and names are inconsistent.")
    if any(fraction < 0.0 for fraction in fractions) or len(set(names)) != len(names):
        raise ValueError("Partition fractions and names must be valid and unique.")
    strata = defaultdict(list)
    for source_id, records in source_records.items():
        if not records:
            raise ValueError(f"Source {source_id!r} has no records.")
        first = records[0]
        key = (
            str(first.get("source_dataset", "unspecified")),
            str(first["event_type"]),
        )
        if any(
            (
                str(record.get("source_dataset", "unspecified")),
                str(record["event_type"]),
            )
            != key
            for record in records
        ):
            raise ValueError(f"Source {source_id!r} crosses stratification labels.")
        strata[key].append(source_id)

    assignment: Dict[str, str] = {}
    rng = random.Random(seed)
    total_sources = sum(len(sources) for sources in strata.values())
    ideal_totals = [total_sources * fraction for fraction in fractions]
    targets = [int(math.floor(value)) for value in ideal_totals]
    for index in sorted(
        range(len(names)),
        key=lambda item: (ideal_totals[item] - targets[item], -item),
        reverse=True,
    )[: total_sources - sum(targets)]:
        targets[index] += 1
    remaining = targets[:]
    for key in sorted(strata):
        sources = sorted(strata[key])
        rng.shuffle(sources)
        ideal = [len(sources) * fraction for fraction in fractions]
        allocated = [0] * len(names)
        for _ in sources:
            candidates = [index for index, capacity in enumerate(remaining) if capacity > 0]
            if not candidates:
                raise RuntimeError("Partition capacity was exhausted unexpectedly.")
            selected = max(
                candidates,
                key=lambda index: (
                    ideal[index] - allocated[index],
                    remaining[index],
                    -index,
                ),
            )
            allocated[selected] += 1
            remaining[selected] -= 1
        cursor = 0
        for name, count in zip(names, allocated):
            for source_id in sources[cursor : cursor + count]:
                assignment[source_id] = name
            cursor += count
    if any(remaining):
        raise RuntimeError("Partition did not meet the global target counts.")
    return assignment
