#!/usr/bin/env python3
# ruff: noqa: T201
"""Compare GuideLLM replay send times against OTEL / WEKA traces.

Reads a ``benchmarks.json`` report (per-request ``info.timings``) and an
optional original JSONL trace. Classifies each successful request:

* ``on_time`` — |request_start - targeted_start| below the threshold
* ``dependency_bound`` — a DAG parent was still running at the target
* ``concurrency_bound`` — slots held by future-timestamp sleeps, or in-flight
  at the observed concurrency cap
* ``server_bound`` — HTTP/backend gap dominates the delay
* ``client_late`` — free to fire (parents done, slots free) but still late

Expected offsets from the trace are matched greedily onto requests sorted by
``relative_timestamp``. Delay classification uses request timings and does
not require a perfect join.

Examples
--------
::

    .venv/bin/python scripts/trace_profile/compare_replay_timings.py \\
      --benchmarks scripts/trace_profile/out/otel_stall_mock.json \\
      --trace otel_stall.jsonl --format otel --backend mock

## WRITTEN BY AI ##
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from guidellm.benchmark import GenerativeBenchmark, GenerativeBenchmarksReport
from guidellm.schemas import GenerativeRequestStats

FormatName = Literal["otel", "weka", "auto"]
ClassName = Literal[
    "on_time",
    "dependency_bound",
    "concurrency_bound",
    "server_bound",
    "client_late",
    "missing_timings",
]
_LLM_OPS = frozenset({"chat", "generate", "text_completion"})
_NON_LLM_OPS = frozenset({"invoke_agent", "execute_tool", "embeddings"})
_FAILED_STATUS = 2
_GAP_IDLE_SECONDS = 5.0
CSV_FIELDS = [
    "backend",
    "format",
    "classification",
    "region",
    "request_id",
    "conversation_id",
    "node_id",
    "parent_node_ids",
    "turn_index",
    "relative_timestamp",
    "trace_conversation",
    "trace_offset",
    "trace_offset_error",
    "targeted_start",
    "dequeued",
    "scheduled_at",
    "resolve_start",
    "request_start",
    "request_end",
    "dispatch_delay_ms",
    "sleep_overshoot_ms",
    "http_gap_ms",
    "service_ms",
    "parent_end",
    "parent_slack_ms",
    "sleeping_future_count",
    "in_flight_count",
    "observed_concurrency_cap",
]


@dataclass
class TraceEvent:
    conversation: str
    offset: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class RequestRow:
    stats: GenerativeRequestStats
    classification: ClassName = "missing_timings"
    region: str = "unknown"
    trace_conversation: str | None = None
    trace_offset: float | None = None
    sleeping_future_count: int = 0
    in_flight_count: int = 0
    observed_concurrency_cap: int = 0
    parent_end: float | None = None


def _parse_otel_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return None


def _span_attrs(span: dict[str, Any]) -> dict[str, Any]:
    attrs = span.get("attributes")
    return attrs if isinstance(attrs, dict) else {}


def _is_llm_span(span: dict[str, Any]) -> bool:
    attrs = _span_attrs(span)
    status = span.get("status")
    if isinstance(status, dict) and status.get("code") == _FAILED_STATUS:
        return False
    op = attrs.get("gen_ai.operation.name")
    if op in _NON_LLM_OPS:
        return False
    if op in _LLM_OPS:
        return True
    return any(
        attrs.get(key) is not None
        for key in (
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.prompt_tokens",
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.completion_tokens",
        )
    )


def _usage_tokens(attrs: dict[str, Any]) -> tuple[int | None, int | None]:
    prompt = attrs.get("gen_ai.usage.input_tokens")
    if prompt is None:
        prompt = attrs.get("gen_ai.usage.prompt_tokens")
    output = attrs.get("gen_ai.usage.output_tokens")
    if output is None:
        output = attrs.get("gen_ai.usage.completion_tokens")
    try:
        prompt_i = int(prompt) if prompt is not None else None
    except (TypeError, ValueError):
        prompt_i = None
    try:
        output_i = int(output) if output is not None else None
    except (TypeError, ValueError):
        output_i = None
    return prompt_i, output_i


def detect_format(path: Path) -> Literal["otel", "weka"]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "spans" in row or "trace_id" in row:
                return "otel"
            if "requests" in row:
                return "weka"
            if "start_time" in row or "attributes" in row:
                return "otel"
            raise ValueError(f"Cannot detect trace format from {path}")
    raise ValueError(f"Empty trace file: {path}")


def load_otel_events(path: Path) -> list[TraceEvent]:
    """Load session-per-line or span-per-line OTEL JSONL into relative offsets."""
    sessions: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row.get("spans"), list):
                trace_id = str(row.get("trace_id") or f"session-{len(order)}")
                sessions.setdefault(trace_id, [])
                if trace_id not in order:
                    order.append(trace_id)
                sessions[trace_id].extend(row["spans"])
                continue
            trace_id = str(row.get("trace_id") or "unknown")
            sessions.setdefault(trace_id, [])
            if trace_id not in order:
                order.append(trace_id)
            sessions[trace_id].append(row)

    events: list[TraceEvent] = []
    for trace_id in order:
        timed: list[tuple[float, dict[str, Any]]] = []
        for span in sessions[trace_id]:
            if not _is_llm_span(span):
                continue
            ts = _parse_otel_timestamp(span.get("start_time"))
            if ts is None:
                continue
            timed.append((ts, span))
        if not timed:
            continue
        origin = min(ts for ts, _ in timed)
        for ts, span in sorted(timed, key=lambda item: item[0]):
            prompt, output = _usage_tokens(_span_attrs(span))
            events.append(
                TraceEvent(
                    conversation=trace_id,
                    offset=ts - origin,
                    prompt_tokens=prompt,
                    output_tokens=output,
                )
            )
    return events


def _flatten_weka_requests(requests: list[Any]) -> list[dict[str, Any]]:
    """Walk WEKA request lists, including declared subagent groups."""
    flat: list[dict[str, Any]] = []
    for item in requests:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "subagent":
            inner = item.get("requests")
            if isinstance(inner, list):
                flat.extend(_flatten_weka_requests(inner))
            continue
        if "t" in item or "in" in item:
            flat.append(item)
    return flat


def load_weka_events(path: Path) -> list[TraceEvent]:
    events: list[TraceEvent] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            conv = str(row.get("id") or "unknown")
            requests = row.get("requests")
            if not isinstance(requests, list):
                continue
            flat = _flatten_weka_requests(requests)
            times = [float(req["t"]) for req in flat if req.get("t") is not None]
            if not times:
                continue
            origin = min(times)
            for req in flat:
                if req.get("t") is None:
                    continue
                events.append(
                    TraceEvent(
                        conversation=conv,
                        offset=float(req["t"]) - origin,
                        prompt_tokens=_as_int(req.get("in")),
                        output_tokens=_as_int(req.get("out")),
                    )
                )
    return events


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def load_trace_events(path: Path, format_name: FormatName) -> list[TraceEvent]:
    resolved: Literal["otel", "weka"]
    if format_name == "auto":
        resolved = detect_format(path)
    else:
        resolved = format_name
    if resolved == "otel":
        return load_otel_events(path)
    return load_weka_events(path)


def iter_successful(benchmark: GenerativeBenchmark) -> list[GenerativeRequestStats]:
    return list(benchmark.requests.successful or [])


def _timing(stats: GenerativeRequestStats, name: str) -> float | None:
    return getattr(stats.info.timings, name)


def _relative_timestamp(stats: GenerativeRequestStats) -> float | None:
    return stats.info.settings.relative_timestamp


def match_trace_events(
    rows: Sequence[RequestRow], events: Sequence[TraceEvent]
) -> None:
    """Greedy match by relative_timestamp order; leftover events stay unmatched."""
    pending = sorted(enumerate(events), key=lambda item: item[1].offset)
    by_rel = sorted(
        (row for row in rows if _relative_timestamp(row.stats) is not None),
        key=lambda row: _relative_timestamp(row.stats) or 0.0,
    )
    used: set[int] = set()
    for row in by_rel:
        rel = _relative_timestamp(row.stats)
        if rel is None:
            continue
        best_i: int | None = None
        best_err = float("inf")
        for index, event in pending:
            if index in used:
                continue
            err = abs(event.offset - rel)
            if err < best_err:
                best_err = err
                best_i = index
            # Offsets are sorted; once we have passed rel by a large margin stop.
            if event.offset - rel > 1.0 and best_i is not None:
                break
        if best_i is None:
            continue
        used.add(best_i)
        event = events[best_i]
        row.trace_conversation = event.conversation
        row.trace_offset = event.offset


def _parent_end(
    stats: GenerativeRequestStats, by_node: dict[tuple[str | None, str | None], RequestRow]
) -> float | None:
    conversation = stats.info.conversation_id
    ends: list[float] = []
    for parent_id in stats.info.parent_node_ids:
        parent = by_node.get((conversation, parent_id))
        if parent is None:
            continue
        end = _timing(parent.stats, "request_end") or _timing(parent.stats, "resolve_end")
        if end is not None:
            ends.append(end)
    return max(ends) if ends else None


def _in_flight_and_sleeping(
    target: float,
    current: RequestRow,
    all_rows: Sequence[RequestRow],
) -> tuple[int, int]:
    """Count live HTTP work and dequeued-but-not-sent later-timestamp sleeps."""
    in_flight = 0
    sleeping_future = 0
    current_rel = _relative_timestamp(current.stats)
    for other in all_rows:
        if other is current:
            continue
        dequeued = _timing(other.stats, "dequeued")
        start = _timing(other.stats, "request_start")
        end = _timing(other.stats, "request_end") or _timing(other.stats, "resolve_end")
        other_rel = _relative_timestamp(other.stats)
        if start is not None and start <= target and (end is None or end > target):
            in_flight += 1
            continue
        # Dequeued, not yet sent, later trace offset: holding a slot in sleep.
        if (
            dequeued is not None
            and dequeued <= target
            and (start is None or start > target)
            and current_rel is not None
            and other_rel is not None
            and other_rel > current_rel
        ):
            sleeping_future += 1
    return in_flight, sleeping_future


def _observed_cap(rows: Sequence[RequestRow]) -> int:
    """Max concurrent in-flight HTTP requests over the run."""
    events: list[tuple[float, int]] = []
    for row in rows:
        start = _timing(row.stats, "request_start")
        end = _timing(row.stats, "request_end") or _timing(row.stats, "resolve_end")
        if start is None:
            continue
        events.append((start, 1))
        if end is not None:
            events.append((end, -1))
    events.sort(key=lambda item: (item[0], item[1]))
    current = 0
    peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def classify_rows(
    rows: list[RequestRow],
    *,
    on_time_ms: float,
) -> None:
    by_node = {
        (row.stats.info.conversation_id, row.stats.info.node_id): row for row in rows
    }
    cap = _observed_cap(rows)
    threshold_s = on_time_ms / 1000.0
    for row in rows:
        timings = row.stats.info.timings
        targeted = timings.targeted_start
        request_start = timings.request_start
        resolve_start = timings.resolve_start
        request_end = timings.request_end
        if targeted is None or request_start is None:
            row.classification = "missing_timings"
            continue
        delay = request_start - targeted
        sleep_overshoot = (
            resolve_start - targeted if resolve_start is not None else delay
        )
        http_gap = (
            request_start - resolve_start if resolve_start is not None else 0.0
        )
        parent_end = _parent_end(row.stats, by_node)
        row.parent_end = parent_end
        in_flight, sleeping_future = _in_flight_and_sleeping(targeted, row, rows)
        row.in_flight_count = in_flight
        row.sleeping_future_count = sleeping_future
        row.observed_concurrency_cap = cap

        if abs(delay) <= threshold_s:
            row.classification = "on_time"
            continue
        if delay < -threshold_s:
            # Early send is still "on time" for lag analysis.
            row.classification = "on_time"
            continue
        if parent_end is not None and parent_end > targeted:
            row.classification = "dependency_bound"
            continue
        if sleeping_future > 0:
            row.classification = "concurrency_bound"
            continue
        if cap > 0 and in_flight >= cap:
            row.classification = "concurrency_bound"
            continue
        # Server/HTTP path ate the delay: resolve started on time, send waited.
        if http_gap > sleep_overshoot and http_gap > threshold_s:
            row.classification = "server_bound"
            continue
        row.classification = "client_late"


def _region_for(row: RequestRow, sorted_rows: Sequence[RequestRow]) -> str:
    rel = _relative_timestamp(row.stats)
    if rel is None:
        return "unknown"
    prev_rel: float | None = None
    for other in sorted_rows:
        other_rel = _relative_timestamp(other.stats)
        if other_rel is None:
            continue
        if other is row:
            break
        prev_rel = other_rel
    if prev_rel is None:
        return "start"
    gap = rel - prev_rel
    if gap >= _GAP_IDLE_SECONDS:
        return "burst_after_gap"
    if gap <= 0.2:
        return "burst"
    return "steady"


def percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (p / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _ms(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 1000.0


def write_csv(
    path: Path,
    rows: Sequence[RequestRow],
    *,
    backend: str,
    format_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            stats = row.stats
            timings = stats.info.timings
            targeted = timings.targeted_start
            resolve_start = timings.resolve_start
            request_start = timings.request_start
            request_end = timings.request_end
            rel = _relative_timestamp(stats)
            parent_slack = None
            if row.parent_end is not None and targeted is not None:
                parent_slack = _ms(targeted - row.parent_end)
            trace_err = None
            if row.trace_offset is not None and rel is not None:
                trace_err = rel - row.trace_offset
            writer.writerow(
                {
                    "backend": backend,
                    "format": format_name,
                    "classification": row.classification,
                    "region": row.region,
                    "request_id": stats.request_id,
                    "conversation_id": stats.info.conversation_id,
                    "node_id": stats.info.node_id,
                    "parent_node_ids": " ".join(stats.info.parent_node_ids),
                    "turn_index": stats.info.turn_index,
                    "relative_timestamp": rel,
                    "trace_conversation": row.trace_conversation,
                    "trace_offset": row.trace_offset,
                    "trace_offset_error": trace_err,
                    "targeted_start": targeted,
                    "dequeued": timings.dequeued,
                    "scheduled_at": timings.scheduled_at,
                    "resolve_start": resolve_start,
                    "request_start": request_start,
                    "request_end": request_end,
                    "dispatch_delay_ms": _ms(
                        None
                        if targeted is None or request_start is None
                        else request_start - targeted
                    ),
                    "sleep_overshoot_ms": _ms(
                        None
                        if targeted is None or resolve_start is None
                        else resolve_start - targeted
                    ),
                    "http_gap_ms": _ms(
                        None
                        if resolve_start is None or request_start is None
                        else request_start - resolve_start
                    ),
                    "service_ms": _ms(
                        None
                        if request_start is None or request_end is None
                        else request_end - request_start
                    ),
                    "parent_end": row.parent_end,
                    "parent_slack_ms": parent_slack,
                    "sleeping_future_count": row.sleeping_future_count,
                    "in_flight_count": row.in_flight_count,
                    "observed_concurrency_cap": row.observed_concurrency_cap,
                }
            )


def print_summary(rows: Sequence[RequestRow], *, backend: str, format_name: str) -> None:
    print(f"backend={backend} format={format_name} requests={len(rows)}")
    classes = [row.classification for row in rows]
    for name in (
        "on_time",
        "client_late",
        "concurrency_bound",
        "dependency_bound",
        "server_bound",
        "missing_timings",
    ):
        count = classes.count(name)
        if count:
            print(f"  {name}: {count}")

    def _delays(predicate) -> list[float]:
        values: list[float] = []
        for row in rows:
            if not predicate(row):
                continue
            targeted = _timing(row.stats, "targeted_start")
            start = _timing(row.stats, "request_start")
            if targeted is None or start is None:
                continue
            values.append((start - targeted) * 1000.0)
        return values

    def _line(label: str, values: list[float]) -> None:
        if not values:
            print(f"  {label}: n=0")
            return
        p50 = percentile(values, 50)
        p95 = percentile(values, 95)
        p99 = percentile(values, 99)
        mean = statistics.fmean(values)
        print(
            f"  {label}: n={len(values)} mean={mean:.2f}ms "
            f"p50={p50:.2f}ms p95={p95:.2f}ms p99={p99:.2f}ms max={max(values):.2f}ms"
        )

    _line("dispatch_delay all", _delays(lambda row: True))
    _line(
        "dispatch_delay client_late",
        _delays(lambda row: row.classification == "client_late"),
    )
    _line(
        "dispatch_delay burst_after_gap",
        _delays(lambda row: row.region == "burst_after_gap"),
    )
    unmatched = sum(1 for row in rows if row.trace_offset is None)
    if unmatched:
        print(f"  unmatched_to_trace: {unmatched}")
    errors = [
        abs((_relative_timestamp(row.stats) or 0.0) - row.trace_offset)
        for row in rows
        if row.trace_offset is not None and _relative_timestamp(row.stats) is not None
    ]
    if errors:
        print(
            f"  relative_timestamp vs trace_offset abs error: "
            f"mean={statistics.fmean(errors):.6f}s max={max(errors):.6f}s"
        )


def load_benchmarks(path: Path) -> list[GenerativeBenchmark]:
    report = GenerativeBenchmarksReport.load_file(path)
    if not report.benchmarks:
        raise ValueError(f"No benchmarks in {path}")
    return report.benchmarks


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify replay dispatch delay (request_start - targeted_start) "
            "and optionally join OTEL/WEKA trace offsets."
        )
    )
    parser.add_argument("--benchmarks", required=True, type=Path)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument(
        "--format",
        choices=("otel", "weka", "auto"),
        default="auto",
        dest="trace_format",
    )
    parser.add_argument(
        "--backend",
        default="unknown",
        help="Tag written into the CSV (mock or vllm).",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--on-time-ms",
        type=float,
        default=5.0,
        help="Absolute dispatch delay at or below this is on_time (default 5ms).",
    )
    parser.add_argument(
        "--benchmark-index",
        type=int,
        default=0,
        help="Which benchmark in the report to analyze (default 0).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    benchmarks = load_benchmarks(args.benchmarks)
    if args.benchmark_index >= len(benchmarks):
        print(
            f"benchmark index {args.benchmark_index} out of range "
            f"(report has {len(benchmarks)})",
            file=sys.stderr,
        )
        return 2
    benchmark = benchmarks[args.benchmark_index]
    successful = iter_successful(benchmark)
    rows = [RequestRow(stats=stats) for stats in successful]
    format_name = args.trace_format
    if args.trace is not None:
        if format_name == "auto":
            format_name = detect_format(args.trace)
        events = load_trace_events(args.trace, format_name)
        match_trace_events(rows, events)
    elif format_name == "auto":
        format_name = "otel"

    classify_rows(rows, on_time_ms=args.on_time_ms)
    by_rel = sorted(
        rows,
        key=lambda row: (
            _relative_timestamp(row.stats) is None,
            _relative_timestamp(row.stats) or 0.0,
        ),
    )
    for row in rows:
        row.region = _region_for(row, by_rel)

    output = args.output
    if output is None:
        output = args.benchmarks.with_name(args.benchmarks.stem + "_compare.csv")
    write_csv(output, rows, backend=args.backend, format_name=format_name)
    print_summary(rows, backend=args.backend, format_name=format_name)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
