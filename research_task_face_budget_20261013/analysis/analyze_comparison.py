"""Frozen standalone statistics; no solver imports or private suite access."""
import argparse
from collections import Counter, defaultdict
import copy
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import tempfile

METHODS = ("ma_rank1", "ma_rank1_structured", "zhao1_rank1", "zhao2t3_rank1",
           "zhao2t3_rank1_structured")
PAIRS = (("ma_rank1", "ma_rank1_structured"),
         ("zhao2t3_rank1", "zhao2t3_rank1_structured"))
REPS = (0, 1, 2)
EXPECTED_CASES = 32
EXPECTED_TEMPLATES = 16
TIME_BUDGET = 5.0
EPSILONS = ("1/1000", "1/100")
BOOTSTRAP_SEED = 20261013947
BOOTSTRAP_REPETITIONS = 10000
STATUSES = ("CERTIFIED_SAFE", "CERTIFIED_UNSAFE", "INCONSISTENT", "UNKNOWN", "ERROR")
ROOT = Path(__file__).resolve().parents[1]


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_groups(groups):
    if not isinstance(groups, list) or len(groups) != EXPECTED_CASES:
        raise ValueError("evaluation_groups must contain exactly 32 rows")
    out, templates = [], defaultdict(list)
    for index, row in enumerate(groups):
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError("Invalid evaluation group row")
        if (type(row.get("template_id")) is not int or type(row.get("structured")) is not bool or
                type(row.get("threshold_policy")) is not int or row["threshold_policy"] not in range(4)):
            raise ValueError("Invalid template_id/structured/threshold_policy metadata")
        epsilon = str(Fraction(str(row["epsilon"])))
        if epsilon not in EPSILONS:
            raise ValueError("Unexpected response epsilon")
        item = dict(row, case_index=index, epsilon=epsilon,
                    population="structural" if row["structured"] else "ordinary")
        out.append(item)
        templates[item["template_id"]].append(item)
    if len({row["name"] for row in out}) != EXPECTED_CASES or len(templates) != EXPECTED_TEMPLATES:
        raise ValueError("Expected 32 unique cases and 16 unique templates")
    for rows in templates.values():
        if len(rows) != 2 or {row["epsilon"] for row in rows} != set(EPSILONS):
            raise ValueError("Each template must retain both response budgets")
        if len({(row["population"], row.get("family"), row.get("threshold_policy")) for row in rows}) != 1:
            raise ValueError("Template metadata differ between response budgets")
    if Counter(row["population"] for row in out) != {"ordinary": 16, "structural": 16}:
        raise ValueError("Expected 16 ordinary and 16 structural cases")
    if Counter(row["threshold_policy"] for row in out) != {policy: 8 for policy in range(4)}:
        raise ValueError("Each threshold policy must have four templates/eight cases")
    return out


def normalize_records(records, groups):
    issues, buckets = [], defaultdict(list)
    for line, row in enumerate(records, 1):
        if not isinstance(row, dict):
            issues.append(f"record {line}: object required")
            continue
        key = (row.get("case_index"), row.get("method"), row.get("rep"))
        index, method, rep = key
        if (type(index) is not int or index not in range(EXPECTED_CASES) or
                method not in METHODS or type(rep) is not int or rep not in REPS):
            issues.append(f"record {line}: unexpected case/method/rep")
            continue
        buckets[key].append((line, row))
    normalized = {}
    for group in groups:
        for method in METHODS:
            for rep in REPS:
                key = (group["case_index"], method, rep)
                found = buckets[key]
                item = dict(case_index=key[0], name=group["name"], method=method, rep=rep,
                            status="MISSING", valid=False, resolved=False, within_budget=False,
                            quick_resolved=False, elapsed=None, overshoot_seconds=None,
                            frontend_calls_total=None, future_vertices_total=None)
                if not found:
                    issues.append(f"missing slot {key}")
                elif len(found) != 1:
                    item["status"] = "INVALID"
                    issues.append(f"duplicate slot {key}: {len(found)} records")
                else:
                    line, row = found[0]
                    result = row.get("result")
                    problems = []
                    if not isinstance(result, dict):
                        result = {}
                        problems.append("result object required")
                    status = result.get("status")
                    elapsed, budget = row.get("elapsed"), row.get("budget_seconds")
                    if status not in STATUSES:
                        problems.append("unknown result status")
                    if row.get("name") != group["name"] or result.get("method") != method:
                        problems.append("case/result method identity mismatch")
                    if not finite_number(elapsed) or elapsed < 0:
                        problems.append("invalid elapsed")
                    if not finite_number(budget) or budget != TIME_BUDGET:
                        problems.append("time budget must equal 5 seconds")
                    resolved = status in ("CERTIFIED_UNSAFE", "INCONSISTENT") or (
                        status == "CERTIFIED_SAFE" and result.get("nonempty_witness") is not None)
                    within = resolved and finite_number(elapsed) and elapsed <= TIME_BUDGET
                    if type(row.get("resolved")) is not bool or row["resolved"] != resolved:
                        problems.append("resolved flag disagrees with status/witness")
                    if type(row.get("resolved_within_budget")) is not bool or row["resolved_within_budget"] != within:
                        problems.append("resolved_within_budget flag disagrees")
                    overshoot = row.get("overshoot_seconds")
                    if (not finite_number(overshoot) or not finite_number(elapsed) or
                            abs(overshoot - max(0.0, elapsed-TIME_BUDGET)) > 1e-9):
                        problems.append("overshoot disagrees with elapsed")
                    quick = result.get("quick_resolved", False)
                    gate = row.get("quick")
                    if type(quick) is not bool or (quick and (status != "CERTIFIED_SAFE" or
                            not isinstance(gate, dict) or gate.get("status") != "QUICK_CERTIFIED_SAFE")):
                        problems.append("invalid quick flag/gate")
                    if not isinstance(row.get("stages"), list) or (gate is not None and not isinstance(gate, dict)):
                        problems.append("invalid stages/quick schema")
                    for field in ("frontend_calls_total", "future_vertices_total"):
                        if type(row.get(field)) is not int or row[field] < 0:
                            problems.append("invalid " + field)
                    if problems:
                        item["status"] = "INVALID"
                        issues.append(f"record {line}: " + "; ".join(problems))
                    else:
                        item.update(status=status, valid=True, resolved=resolved, within_budget=within,
                                    quick_resolved=quick, elapsed=elapsed, overshoot_seconds=overshoot,
                                    frontend_calls_total=row["frontend_calls_total"],
                                    future_vertices_total=row["future_vertices_total"])
                normalized[key] = item
    return normalized, issues


def group_definitions(groups):
    definitions = [("all", lambda row: True)]
    definitions += [("population=" + population, lambda row, p=population: row["population"] == p)
                    for population in ("ordinary", "structural")]
    definitions += [("epsilon=" + epsilon, lambda row, e=epsilon: row["epsilon"] == e) for epsilon in EPSILONS]
    definitions += [(f"population={population};epsilon={epsilon}",
                     lambda row, p=population, e=epsilon: row["population"] == p and row["epsilon"] == e)
                    for population in ("ordinary", "structural") for epsilon in EPSILONS]
    definitions += [(f"threshold_policy={policy}", lambda row, p=policy: row["threshold_policy"] == p)
                    for policy in range(4)]
    return {name: {row["case_index"] for row in groups if test(row)} for name, test in definitions}


def percentile(values, p):
    position = (len(values)-1)*p
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper]-values[lower])*(position-lower)


def cluster_bootstrap(eligible, template_ids, draws):
    logs = defaultdict(list)
    for row in eligible:
        logs[row["template_id"]].append(row["log_speedup"])
    values, sample_counts = [], []
    for draw in draws:
        sample = [value for template in draw for value in logs.get(template, ())]
        if sample:
            values.append(statistics.median(sample))
            sample_counts.append(len(sample))
    values.sort()
    return dict(eligible_cases=len(eligible), eligible_templates=len(logs),
                template_clusters_in_sampling_frame=len(template_ids),
                bootstrap_repetitions=len(draws), nonempty_bootstrap_samples=len(values),
                empty_bootstrap_samples=len(draws)-len(values),
                bootstrap_eligible_cases_min=min(sample_counts) if sample_counts else 0,
                bootstrap_eligible_cases_max=max(sample_counts) if sample_counts else 0,
                paired_log_median_speedup=math.exp(statistics.median(
                    [row["log_speedup"] for row in eligible])) if eligible else None,
                ci95_low=math.exp(percentile(values, .025)) if values else None,
                ci95_high=math.exp(percentile(values, .975)) if values else None,
                ci_conditioned_on_nonempty=bool(len(values) != len(draws)))


def analyze(records, raw_groups):
    groups = validate_groups(raw_groups)
    runs, issues = normalize_records(records, groups)
    case_rows, case_map = [], {}
    for group in groups:
        for method in METHODS:
            rows = [runs[(group["case_index"], method, rep)] for rep in REPS]
            statuses = {row["status"] for row in rows}
            success = all(row["valid"] and row["within_budget"] for row in rows) and len(statuses) == 1
            times = [row["elapsed"] for row in rows if row["elapsed"] is not None]
            row = dict(case_index=group["case_index"], name=group["name"], template_id=group["template_id"],
                       population=group["population"], epsilon=group["epsilon"], method=method,
                       expected_repetitions=3, valid_repetitions=sum(item["valid"] for item in rows),
                       within_budget_repetitions=sum(item["within_budget"] for item in rows),
                       late_resolved_repetitions=sum(item["resolved"] and not item["within_budget"] for item in rows),
                       success=success, conclusion=next(iter(statuses)) if success else None,
                       all_quick=all(item["quick_resolved"] for item in rows),
                       median_elapsed=statistics.median(times) if len(times) == 3 else None,
                       status_rep0=rows[0]["status"], status_rep1=rows[1]["status"], status_rep2=rows[2]["status"])
            case_rows.append(row)
            case_map[(group["case_index"], method)] = row
    paired_rows = []
    for baseline, candidate in PAIRS:
        for group in groups:
            base = case_map[(group["case_index"], baseline)]
            cand = case_map[(group["case_index"], candidate)]
            reasons = []
            if not base["success"]:
                reasons.append("baseline_not_stable_within_budget")
            if not cand["success"]:
                reasons.append("candidate_not_stable_within_budget")
            if base["success"] and cand["success"] and base["conclusion"] != cand["conclusion"]:
                reasons.append("conclusion_mismatch")
            common_quick = base["all_quick"] and cand["all_quick"]
            if common_quick:
                reasons.append("common_quick")
            if ((base["median_elapsed"] is not None and base["median_elapsed"] <= 0) or
                    (cand["median_elapsed"] is not None and cand["median_elapsed"] <= 0)):
                reasons.append("nonpositive_median_time")
            eligible = not reasons
            ratio = base["median_elapsed"]/cand["median_elapsed"] if eligible else None
            paired_rows.append(dict(case_index=group["case_index"], name=group["name"],
                template_id=group["template_id"], population=group["population"], epsilon=group["epsilon"],
                baseline=baseline, candidate=candidate, baseline_success=base["success"],
                candidate_success=cand["success"], baseline_conclusion=base["conclusion"],
                candidate_conclusion=cand["conclusion"], common_quick=common_quick, timing_eligible=eligible,
                exclusion_reasons=";".join(reasons), new_success=not base["success"] and cand["success"],
                lost_success=base["success"] and not cand["success"], baseline_median=base["median_elapsed"],
                candidate_median=cand["median_elapsed"], speedup_ratio=ratio,
                log_speedup=math.log(ratio) if eligible else None,
                candidate_slowdown_fraction=cand["median_elapsed"]/base["median_elapsed"]-1 if eligible else None,
                slowdown_gt_5pct=eligible and cand["median_elapsed"] > 1.05*base["median_elapsed"]))
    template_ids = sorted({row["template_id"] for row in groups})
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [[template_ids[rng.randrange(EXPECTED_TEMPLATES)] for _ in template_ids]
             for _ in range(BOOTSTRAP_REPETITIONS)]
    group_sets = group_definitions(groups)
    method_summaries, pair_summaries = [], []
    for group_name, indices in group_sets.items():
        for method in METHODS:
            rows = [runs[(index, method, rep)] for index in sorted(indices) for rep in REPS]
            observed = [row for row in records if isinstance(row, dict) and
                        type(row.get("case_index")) is int and row["case_index"] in indices and
                        row.get("method") == method]
            costs = [row["elapsed"] for row in observed if finite_number(row.get("elapsed")) and row["elapsed"] >= 0]
            statuses = Counter(row["status"] for row in rows)
            method_summaries.append(dict(group=group_name, method=method, expected_cases=len(indices),
                successful_cases=sum(case_map[(index, method)]["success"] for index in indices),
                expected_runs=len(rows), observed_records=len(observed),
                resolved_within_budget_runs=sum(row["within_budget"] for row in rows),
                late_resolved_runs=sum(row["resolved"] and not row["within_budget"] for row in rows),
                total_observed_elapsed_seconds=math.fsum(costs),
                median_observed_elapsed_seconds=statistics.median(costs) if costs else None,
                invalid_elapsed_records=len(observed)-len(costs),
                **{status.lower()+"_runs": statuses[status] for status in (*STATUSES, "MISSING", "INVALID")}))
        for baseline, candidate in PAIRS:
            rows = [row for row in paired_rows if row["baseline"] == baseline and row["case_index"] in indices]
            eligible = [row for row in rows if row["timing_eligible"]]
            pair_summaries.append(dict(group=group_name, baseline=baseline, candidate=candidate,
                expected_cases=len(indices), baseline_successes=sum(row["baseline_success"] for row in rows),
                candidate_successes=sum(row["candidate_success"] for row in rows),
                new_successes=sum(row["new_success"] for row in rows), lost_successes=sum(row["lost_success"] for row in rows),
                common_quick_cases=sum(row["common_quick"] for row in rows),
                slowdowns_gt_5pct=sum(row["slowdown_gt_5pct"] for row in rows),
                **cluster_bootstrap(eligible, template_ids, draws)))
    conflicts, nesting = [], []
    by_case = defaultdict(list)
    for row in runs.values():
        if row["valid"] and row["resolved"]:
            by_case[row["case_index"]].append(row)
    for group in groups:
        evidence = by_case[group["case_index"]]
        if len({row["status"] for row in evidence}) > 1:
            conflicts.append(dict(case_index=group["case_index"], name=group["name"],
                conclusions=";".join(sorted({row["status"] for row in evidence})),
                evidence=";".join(f'{row["method"]}/rep{row["rep"]}/{row["status"]}' for row in evidence)))
    for template in template_ids:
        lower = next(row for row in groups if row["template_id"] == template and row["epsilon"] == EPSILONS[0])
        upper = next(row for row in groups if row["template_id"] == template and row["epsilon"] == EPSILONS[1])
        for small in by_case[lower["case_index"]]:
            for large in by_case[upper["case_index"]]:
                if ((large["status"] == "CERTIFIED_SAFE" and small["status"] == "CERTIFIED_UNSAFE") or
                        (large["status"] == "INCONSISTENT" and small["status"] in ("CERTIFIED_SAFE", "CERTIFIED_UNSAFE"))):
                    nesting.append(dict(template_id=template, small_case=lower["name"], large_case=upper["name"],
                        small_method=small["method"], small_rep=small["rep"], small_status=small["status"],
                        large_method=large["method"], large_rep=large["rep"], large_status=large["status"]))
    valid_costs = [row["elapsed"] for row in records if isinstance(row, dict) and
                   finite_number(row.get("elapsed")) and row["elapsed"] >= 0]
    tables = dict(run_slots=list(runs.values()), case_methods=case_rows, paired_cases=paired_rows,
                  method_groups=method_summaries, paired_groups=pair_summaries,
                  slowdowns_gt_5pct=[row for row in paired_rows if row["slowdown_gt_5pct"]],
                  success_changes=[row for row in paired_rows if row["new_success"] or row["lost_success"]],
                  conclusion_conflicts=conflicts, nested_budget_contradictions=nesting,
                  validation_issues=[dict(issue_index=i+1, message=message) for i, message in enumerate(issues)])
    summary = dict(schema="taskface_statistics_v1", expected_cases=32, expected_methods=5,
        expected_repetitions=3, expected_runs=480, observed_records=len(records),
        statistics_valid_for_reporting=len(records) == 480 and not issues and not conflicts and not nesting,
        input_complete_and_valid=len(records) == 480 and not issues, validation_issues=issues,
        valid_runs=sum(row["valid"] for row in runs.values()),
        missing_runs=sum(row["status"] == "MISSING" for row in runs.values()),
        invalid_slots=sum(row["status"] == "INVALID" for row in runs.values()),
        total_observed_elapsed_seconds=math.fsum(valid_costs), cost_records_with_valid_elapsed=len(valid_costs),
        cost_records_with_invalid_elapsed=len(records)-len(valid_costs),
        cost_interpretation="All observed runtimes, including UNKNOWN, ERROR and late results; unmatched totals are not same-task speedup.",
        bootstrap=dict(seed=BOOTSTRAP_SEED, repetitions=BOOTSTRAP_REPETITIONS, template_clusters=16,
            algorithm="For each eligible case log(median_baseline/median_candidate); median logs then exp. Draw 16 templates with replacement, retain both budget cases for each occurrence, apply frozen group/eligibility filters, median retained logs then exp. Percentile 95% CI uses linear interpolation on nonempty log statistics; empty draws counted explicitly.",
            seed_scope="One shared 10000-draw template sequence for every pair and group"),
        method_groups=method_summaries, paired_groups=pair_summaries,
        slowdowns_gt_5pct=tables["slowdowns_gt_5pct"], success_changes=tables["success_changes"],
        conclusion_conflicts=conflicts, nested_budget_contradictions=nesting)
    return summary, tables


def write_outputs(output, summary, tables):
    output.mkdir(parents=True, exist_ok=True)
    (output/"SUMMARY_STATISTICS.json").write_text(json.dumps(summary, ensure_ascii=False,
        indent=2, allow_nan=False)+"\n", encoding="utf-8")
    empty_headers = {"conclusion_conflicts": ["case_index", "name", "conclusions", "evidence"],
        "nested_budget_contradictions": ["template_id", "small_case", "large_case", "small_method", "small_rep", "small_status", "large_method", "large_rep", "large_status"],
        "validation_issues": ["issue_index", "message"]}
    for name, rows in tables.items():
        fields = list(rows[0]) if rows else empty_headers.get(name, list(tables["paired_cases"][0]))
        with (output/(name+".csv")).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def mock_data():
    groups, records = [], []
    for index in range(32):
        groups.append(dict(name=f"mock_{index:03d}", template_id=index//2,
            structured=(index//2) % 4 >= 2, family="mock", epsilon=EPSILONS[index % 2], threshold_policy=index//8))
        for method in METHODS:
            for rep in REPS:
                elapsed = 2.0 if method.endswith("structured") else 4.0
                quick = index == 0
                if quick:
                    elapsed = .01
                records.append(dict(case_index=index, name=groups[-1]["name"], method=method, rep=rep,
                    result=dict(method=method, status="CERTIFIED_SAFE", nonempty_witness={"mock": True},
                                quick_resolved=quick), elapsed=elapsed, resolved=True, resolved_within_budget=True,
                    budget_seconds=5.0, overshoot_seconds=0.0, stages=[],
                    quick=dict(status="QUICK_CERTIFIED_SAFE") if quick else dict(status="QUICK_NOT_RESOLVED"),
                    frontend_calls_total=1, future_vertices_total=2))
    return records, groups


def self_test():
    records, groups = mock_data()
    def row(index, method, rep):
        return next(item for item in records if (item["case_index"], item["method"], item["rep"]) == (index, method, rep))
    def change(index, method, status, elapsed, rep=None):
        for repetition in REPS if rep is None else (rep,):
            item = row(index, method, repetition)
            item["result"]["status"] = status
            item["elapsed"] = elapsed
            item["resolved"] = status != "UNKNOWN" and status != "ERROR"
            item["resolved_within_budget"] = item["resolved"] and elapsed <= 5
            item["overshoot_seconds"] = max(0, elapsed-5)
    change(1, "ma_rank1_structured", "CERTIFIED_SAFE", 4.4)
    change(2, "ma_rank1_structured", "CERTIFIED_SAFE", 5.1, 0)
    change(3, "ma_rank1", "UNKNOWN", 5.2)
    change(4, "ma_rank1_structured", "ERROR", .2)
    change(5, "zhao1_rank1", "CERTIFIED_UNSAFE", 1.0)
    change(6, "ma_rank1", "CERTIFIED_UNSAFE", 1.0)
    change(8, "ma_rank1_structured", "CERTIFIED_UNSAFE", 2.0, 0)
    summary, tables = analyze(records, groups)
    assert summary["input_complete_and_valid"] and summary["expected_runs"] == len(records) == 480
    assert summary["total_observed_elapsed_seconds"] == math.fsum(item["elapsed"] for item in records)
    assert summary["conclusion_conflicts"] and summary["nested_budget_contradictions"]
    paired = {(item["case_index"], item["baseline"]): item for item in tables["paired_cases"]}
    assert paired[(0, "ma_rank1")]["exclusion_reasons"] == "common_quick"
    assert paired[(1, "ma_rank1")]["slowdown_gt_5pct"]
    assert paired[(2, "ma_rank1")]["lost_success"] and not paired[(2, "ma_rank1")]["timing_eligible"]
    assert paired[(3, "ma_rank1")]["new_success"] and paired[(4, "ma_rank1")]["lost_success"]
    assert not paired[(6, "ma_rank1")]["timing_eligible"] and not paired[(8, "ma_rank1")]["timing_eligible"]
    ratio = next(item for item in tables["paired_groups"] if item["group"] == "all" and item["baseline"] == "zhao2t3_rank1")
    assert ratio["eligible_cases"] == 31 and ratio["paired_log_median_speedup"] == 2.0
    assert ratio["ci95_low"] == ratio["ci95_high"] == 2.0
    rng = random.Random(BOOTSTRAP_SEED)
    draws = [[rng.randrange(16) for _ in range(16)] for _ in range(10000)]
    direct = cluster_bootstrap([dict(template_id=0, log_speedup=math.log(2)),
        dict(template_id=0, log_speedup=math.log(8))], list(range(16)), draws)
    assert direct["paired_log_median_speedup"] == 4.0 and direct["empty_bootstrap_samples"] > 0
    assert direct["bootstrap_eligible_cases_min"] % 2 == direct["bootstrap_eligible_cases_max"] % 2 == 0
    empty = cluster_bootstrap([], list(range(16)), draws)
    assert empty["empty_bootstrap_samples"] == 10000 and empty["ci95_low"] is None
    assert direct["eligible_templates"] == 1 and empty["eligible_templates"] == 0
    assert len(group_definitions(validate_groups(groups))) == 13
    assert all(item["expected_cases"] == 8 for item in tables["method_groups"] if item["group"].startswith("threshold_policy="))
    assert next(item for item in tables["method_groups"] if item["group"] == "all" and item["method"] == "ma_rank1")["median_observed_elapsed_seconds"] == 4.0
    missing, issues = normalize_records(records[:-1], validate_groups(groups))
    assert sum(item["status"] == "MISSING" for item in missing.values()) == 1 and issues
    duplicate, issues = normalize_records(records+[copy.deepcopy(records[0])], validate_groups(groups))
    assert sum(item["status"] == "INVALID" for item in duplicate.values()) == 1 and issues
    bad = copy.deepcopy(records)
    bad[0]["resolved_within_budget"] = False
    invalid, issues = normalize_records(bad, validate_groups(groups))
    assert not invalid[(0, METHODS[0], 0)]["valid"] and issues
    with tempfile.TemporaryDirectory(prefix="stats_mock_", dir=Path(__file__).parent) as directory:
        write_outputs(Path(directory), summary, tables)
        assert json.loads((Path(directory)/"SUMMARY_STATISTICS.json").read_text(encoding="utf-8"))["expected_runs"] == 480
        with (Path(directory)/"run_slots.csv").open(encoding="utf-8", newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 480
    return dict(status="PASS", mock_cases=32, mock_runs=480, bootstrap_repetitions=10000,
        checks=["full_denominators_and_total_cost", "quick_exclusion", "strict_slowdown", "new_lost_success",
                "late_unknown_error_and_unstable", "conflicts_and_nested_budgets", "paired_log_ratio",
                "template_cluster_and_empty_draws", "missing_duplicate_flag_validation", "json_csv_roundtrip"],
        real_formal_data_read=False, formal_suite_generated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=ROOT/"results"/"formal_main"/"records.jsonl")
    parser.add_argument("--groups", type=Path, default=ROOT/"protocol"/"frozen_suite"/"evaluation_groups.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    records_bytes, groups_bytes = args.records.read_bytes(), args.groups.read_bytes()
    records = [json.loads(line) for line in records_bytes.decode("utf-8-sig").splitlines() if line.strip()]
    summary, tables = analyze(records, json.loads(groups_bytes.decode("utf-8-sig")))
    summary["source_sha256"] = dict(records=hashlib.sha256(records_bytes).hexdigest(),
        evaluation_groups=hashlib.sha256(groups_bytes).hexdigest(), script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_outputs(args.output, summary, tables)
    print(json.dumps(dict(output=str(args.output.resolve()), input_complete_and_valid=summary["input_complete_and_valid"],
        statistics_valid_for_reporting=summary["statistics_valid_for_reporting"], expected_runs=480, observed_records=len(records))))
    if not summary["input_complete_and_valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
