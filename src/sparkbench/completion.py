"""Offline analysis of a dated evidence-completion experiment.

Inputs are public scalar observations, not production credentials or request bodies.
Each attempt remains a row. The registered attempt selects a comparison, but does
not change historical outcomes. All rates use source usage and explicit windows.
"""
import csv
import json
import math
import statistics
from pathlib import Path


def quantile(values, probability):
    values = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not values:
        return None
    position = (len(values) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def stats(values):
    values = [x for x in values if x is not None and math.isfinite(float(x))]
    return {"n": len(values), "min": min(values) if values else None,
            "median": quantile(values, .5), "p95": quantile(values, .95),
            "max": max(values) if values else None}


def valid_response(row):
    return (row.get("result_state") == "received" and row.get("http_status") == 200
            and row.get("done") is True and row.get("finish_reason") in ("stop", "length", "tool_calls")
            and not row.get("integrity_flags") and row.get("origin_input_tokens") is not None
            and row.get("origin_output_tokens") is not None)


def observed_rejection(row):
    return (row.get('result_state')=='local_rejection' and not row.get('attempt_started')
            and row.get('offered_recorded') is True
            and row.get('admission_outcome') in ('queue_full','queue_expired'))


def counter_window(before, after, rows):
    """Require drained boundaries, monotonic counters, and source usage conservation."""
    names = ("requests", "input_tokens", "output_tokens", "cached_tokens")
    issues = []
    if not before or not after:
        return {"valid": False, "issues": ["missing_boundary"], "delta": {}, "cache_fraction": None}
    if any(boundary.get(key) != 0 for boundary in (before, after) for key in ("running", "queued")):
        issues.append("boundary_not_idle")
    delta = {}
    for name in names:
        if 'series' in before or 'series' in after:
            left, right = before.get('series', {}).get(name, {}), after.get('series', {}).get(name, {})
            if set(left) != set(right) or not left:
                issues.append('counter_series_changed:' + name)
            elif any(right[key] < left[key] for key in left):
                issues.append('counter_series_reset:' + name)
        if before.get(name) is None or after.get(name) is None:
            issues.append("missing_counter:" + name)
        else:
            delta[name] = after[name] - before[name]
            if delta[name] < 0:
                issues.append("counter_reset:" + name)
    if delta.get("requests") != len(rows):
        issues.append("request_count_mismatch")
    if not all(valid_response(row) for row in rows):
        issues.append("response_or_source_usage_incomplete")
    else:
        for counter, field in (("input_tokens", "origin_input_tokens"), ("output_tokens", "origin_output_tokens")):
            if delta.get(counter) != sum(row[field] for row in rows):
                issues.append("usage_mismatch:" + counter)
    if delta.get("cached_tokens", 0) > delta.get("input_tokens", 0):
        issues.append("cache_exceeds_input")
    valid = not issues
    return {"valid": valid, "issues": issues, "delta": delta,
            "cache_fraction": delta["cached_tokens"] / delta["input_tokens"]
            if valid and delta.get("input_tokens", 0) > 0 else None}


def pair_analysis(pair, blocks, requests):
    arms = {}
    for mode in ("isolated", "mixed"):
        candidates = [b for b in blocks if b.get("selected") and b.get("kind") == "interference"
                      and b.get("pair") == pair and b.get("mode") == mode]
        if len(candidates) != 1:
            arms[mode] = {"complete": False, "observed": False,"reason": "missing_or_duplicate_arm", "short_rows": []}
            continue
        block = candidates[0]
        short = [r for r in requests if r.get("selected") and r["block_id"] == block["id"]
                 and r.get("stream_kind") == "short"]
        complete_rows = [r for r in short if valid_response(r) and r.get("arrival_to_visible_s") is not None]
        all_rows=[r for r in requests if r.get('selected') and r['block_id']==block['id']]
        rejections=[r for r in all_rows if observed_rejection(r)]
        observed=(len(all_rows)==block.get('planned') and block.get('offered')==block.get('planned')
                  and len(rejections)==block.get('local_rejections',0)
                  and all(valid_response(r) or observed_rejection(r) for r in all_rows)
                  and all(r.get('upstream_max_tokens')==r.get('output_cap') for r in all_rows if valid_response(r))
                  and block.get('source_window',{}).get('valid')
                  and not block.get('thermal_interrupted') and not block.get('stop'))
        valid = (len(short) == 32 and len(complete_rows) == 32 and block.get("offered") == block.get("planned")
                 and block.get("local_rejections", 0) == 0 and block.get("source_window", {}).get("valid")
                 and all(r.get('upstream_max_tokens')==r.get('output_cap') for r in complete_rows)
                 and not block.get("thermal_interrupted") and not block.get("stop"))
        arms[mode] = {"complete": bool(valid),"observed":bool(observed), "offered": block.get("offered"), "planned": block.get("planned"),
                      "local_rejections": block.get("local_rejections", 0), "short_complete": len(complete_rows),
                      "short_planned": 32, "wait_s": stats([r["arrival_to_visible_s"] for r in complete_rows]),
                      "e2e_s": stats([r.get("arrival_to_done_s") for r in complete_rows]),
                      "output_tokens": stats([r.get("origin_output_tokens") for r in complete_rows]),
                      "block_id": block["id"], "short_rows": short}
    isolated = {r["arrival_index"]: r for r in arms["isolated"]["short_rows"]}
    mixed = {r["arrival_index"]: r for r in arms["mixed"]["short_rows"]}
    identities_match = (set(isolated) == set(mixed) == set(range(32))
                        and all(isolated[i]["prompt_sha256"] == mixed[i]["prompt_sha256"]
                                and isolated[i]["planned_offset_s"] == mixed[i]["planned_offset_s"]
                                and isolated[i]["output_cap"] == mixed[i]["output_cap"] for i in isolated))
    qualified = arms["isolated"]["complete"] and arms["mixed"]["complete"] and identities_match
    ratio = None
    differences = []
    if qualified:
        a, b = arms["isolated"]["wait_s"], arms["mixed"]["wait_s"]
        ratio = {key: b[key] / a[key] if a[key] > 0 else None for key in ("median", "p95")}
        differences = [mixed[i]["arrival_to_visible_s"] - isolated[i]["arrival_to_visible_s"] for i in range(32)]
    for arm in arms.values():
        arm.pop("short_rows")
    observed=arms['isolated']['observed'] and arms['mixed']['observed'] and identities_match
    return {"pair": pair, "qualified": bool(qualified),"observed":bool(observed), "identities_match": identities_match, "arms": arms,
            "mixed_over_isolated": ratio, "paired_wait_difference_s": stats(differences)}


def analyze(data):
    requests, blocks = data["requests"], data["blocks"]
    if len({r["case_id"] for r in requests}) != len(requests):
        raise ValueError("Duplicate attempt identity")
    analyzed_blocks = []
    for block in blocks:
        rows = [r for r in requests if r["block_id"] == block["id"] and r.get("attempt_started")]
        window = counter_window(block.get("before"), block.get("after"), rows)
        if block.get("thermal_interrupted") or block.get("stop"):
            window.update(valid=False, cache_fraction=None)
            window["issues"].append("interrupted_block")
        duration = (block.get("end_epoch") or 0) - (block.get("start_epoch") or 0)
        analyzed_blocks.append({**block, "source_window": window,
                                "source_output_tokens_s": sum(r["origin_output_tokens"] for r in rows) / duration
                                if duration > 0 and window["valid"] else None})
    selected = [r for r in requests if r.get("selected")]
    lengths = []
    for row in selected:
        if row.get("stage") != "public" or row.get("kind") != "length":
            continue
        block = next((b for b in analyzed_blocks if b["id"] == row["block_id"]), {})
        target=row.get('target_input_tokens')
        calibrated_input=(target is not None and row.get('origin_input_tokens') is not None
                          and abs(row['origin_input_tokens']-target)<=max(32,.1*target))
        cap_forwarded=row.get('upstream_max_tokens')==row.get('output_cap')
        good = valid_response(row) and block.get("source_window", {}).get("valid", False) and calibrated_input and cap_forwarded
        lengths.append({**row, "qualified_response": bool(good),
                        "input_calibrated":calibrated_input,"output_cap_forwarded":cap_forwarded,
                        "output_fraction": row["origin_output_tokens"] / row["output_cap"]
                        if good and row.get("output_cap", 0) > 0 else None,
                        "fixed_output_qualified": bool(good and row.get("upstream_ignore_eos") is True
                                                       and row["origin_output_tokens"] == row["output_cap"])})
    pairs = [pair_analysis(i, analyzed_blocks, requests) for i in range(5)]
    cache = []
    for pair in range(3):
        arms = {}
        for role in ("seed", "probe"):
            bs = [b for b in analyzed_blocks if b.get("selected") and b.get("stage") == "cache"
                  and b.get("pair") == pair and b.get("role") == role]
            arms[role] = bs[0] if len(bs) == 1 else {"source_window": {"valid": False}}
        cache.append({"pair": pair, "qualified": all(b["source_window"]["valid"] for b in arms.values()),
                      "arms": {role: {"block_id": b.get("id"), **b["source_window"]} for role, b in arms.items()}})
    monitored = [r for r in data.get("telemetry", []) if r.get('in_measurement_window', True)]
    node_stats = {}
    for rank in (0, 1):
        rows = [r for r in monitored if r["node"] == rank]
        node_stats[str(rank)] = {name: stats([r.get(name) for r in rows]) for name in
                                ("temperature_c", "gpu_util_pct", "gpu_clock_mhz", "power_w", "memory_available_gib")}
    complete = [r for r in selected if valid_response(r)]
    selected_started = [r for r in selected if r.get("attempt_started")]
    rejected = [r for r in selected if observed_rejection(r)]
    started = [r for r in requests if r.get("attempt_started")]
    unbilled = [r["case_id"] for r in started if r.get("cost_usd") is None]
    expected_ids = [r['logical_case_id'] for r in selected]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError('More than one selected attempt for a logical case')
    return {"schema_version": 1, "date": "2026-10-19", "historical_results_pooled": False,
            "coverage": {"planned_logical_requests": 695, "selected_attempt_rows": len(selected),
                         "selected_attempts_started": len(selected_started), "selected_source_complete": len(complete),
                         "all_attempts_started": sum(bool(r.get("attempt_started")) for r in requests),
                         "local_rejections_observed":len(rejected),
                         "logical_outcomes_observed":len(complete)+len(rejected),
                         "length_conditions_complete": sum(r["qualified_response"] for r in lengths),
                         "length_conditions_planned": 8, "fixed_output_qualified": sum(r["fixed_output_qualified"] for r in lengths),
                         "interference_pairs_qualified": sum(p["qualified"] for p in pairs), "interference_pairs_planned": 5,
                         "interference_pairs_observed":sum(p['observed'] for p in pairs),
                         "cache_pairs_qualified": sum(p["qualified"] for p in cache), "cache_pairs_planned": 3},
            "lengths": lengths, "interference": pairs, "cache": cache, "nodes": node_stats,
            "rejections":[{key:r.get(key) for key in ['case_id','block_id','pair','mode','stream_kind','admission_outcome','rejection_queue_s']} for r in rejected],
            "blocks": analyzed_blocks,
            "billing": {"started_attempts": len(started), "attributed_attempts": len(started)-len(unbilled),
                        "missing_attempts": unbilled, "complete": not unbilled},
            "cost_usd_all_attempts": sum(r["cost_usd"] for r in started if r.get("cost_usd") is not None),
            "limits": ["Public output caps are not achieved fixed lengths; ignore_eos forwarding is checked per request.",
                       "Five paired blocks are the replication units; request rows are not independent experiments.",
                       "Zero designed prefix overlap is not an assumed zero measured cache fraction.",
                       "Sensor power is not wall-meter energy; no H200 same-workload comparison."]}


def write_summary(experiment, output):
    experiment, output = Path(experiment), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = {name: json.loads((experiment / "data" / (name + ".json")).read_text()) for name in ("requests", "blocks", "telemetry")}
    summary = analyze(data)
    (output / "completion-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for name, rows in (("requests", data["requests"]), ("lengths", summary["lengths"])):
        fields = sorted({k for row in rows for k, value in row.items() if not isinstance(value, (list, dict))})
        with (output / (name + ".csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return summary
