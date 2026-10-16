"""Independent low-cost provenance/schema audit; never imports production/generator code."""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_METHOD_HASH = "5e203a815d35bbb28462d0399dfcec263f64c8a5b8b50509a151488272bdc6f2"
EXPECTED_INPUT_HASH = "bfd1fe58dfa49e24e21d7e01f2be8ee758757bcf96d7170a63bf0380534f9d66"
FIELDS = {"name", "group", "initial", "past", "prior", "packet", "future", "H", "rho", "limit"}
POLICIES = ("public_envelope_plus_5percent_scale", "compatible_reference_plus_1e-2_scale",
            "compatible_reference_plus_1e-4_scale", "compatible_reference_minus_1e-3_scale")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    return json.loads((ROOT/relative).read_text(encoding="utf-8"))


def physical_key(case):
    return json.dumps({key: value for key, value in case.items()
                       if key not in ("name", "group")}, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="qa/GENERATION_FREEZE_AUDIT.json")
    args = parser.parse_args()
    checks, failures = [], []

    def check(name, passed, detail=None):
        item = dict(check=name, passed=bool(passed))
        if detail is not None:
            item["detail"] = detail
        checks.append(item)
        if not passed:
            failures.append(item)

    methods = read("inputs/method_freeze.json")
    frozen = read("inputs/HOLDOUT_FROZEN.json")
    cases = read("inputs/holdout_48.json")
    private = read("qa/holdout_generation_private.json")
    rules = read("inputs/generation_protocol.json")
    check("trusted_method_freeze_hash", digest(ROOT/"inputs/method_freeze.json") == EXPECTED_METHOD_HASH)
    check("trusted_input_hash", digest(ROOT/"inputs/holdout_48.json") == EXPECTED_INPUT_HASH)
    check("manifest_method_reference", frozen["frozen_methods_sha256"] == EXPECTED_METHOD_HASH)
    check("manifest_input_hash", frozen["inputs_sha256"] == digest(ROOT/"inputs/holdout_48.json"))
    check("manifest_private_hash", frozen["private_generation_sha256"] == digest(ROOT/"qa/holdout_generation_private.json"))
    for relative, expected in methods["file_sha256"].items():
        check("frozen_file:"+relative, (ROOT/relative).is_file() and digest(ROOT/relative) == expected)
    check("generator_three_way_hash", frozen["generator_sha256"] == rules["generator_sha256"]
          == digest(ROOT/"implementation/generate_new_holdout.py"))
    check("generation_protocol_three_way_hash", frozen["protocol_sha256"] == rules["full_protocol_sha256"]
          == digest(ROOT/"analysis/NEW_HOLDOUT_PROTOCOL_CN.txt"))
    check("frozen_seed_and_version", frozen["seed"] == private["seed"] == rules["seed"] == 202610120947
          and frozen["version"] == private["version"] == rules["version"])
    check("exact_public_schema", all(set(case) == FIELDS for case in cases))
    check("case_private_bijection", len(cases) == len(private["cases"]) == 48
          and [case["name"] for case in cases] == [case["name"] for case in private["cases"]]
          and len({case["name"] for case in cases}) == 48
          and [item["index"] for item in private["cases"]] == list(range(48)))
    counts = Counter(case["group"] for case in cases)
    expected_groups = dict(holdout_random=32, holdout_four_hold=8,
                           holdout_valley_velocity=4, holdout_valley_position=4)
    check("fixed_group_counts", dict(counts) == expected_groups, dict(counts))
    thresholds = Counter(item["threshold_policy"] for item in private["cases"])
    check("four_thresholds_twelve_each", dict(thresholds) == {policy: 12 for policy in POLICIES}, dict(thresholds))
    new_keys = [physical_key(case) for case in cases]
    check("no_internal_physical_duplicates", len(set(new_keys)) == 48)
    old_keys = {physical_key(case) for relative in ("inputs/development_43.json", "inputs/development_ma48.json")
                for case in read(relative)}
    check("no_copied_development_physical_cases", not set(new_keys).intersection(old_keys))
    expected_methods = ["ma_strip", "ma_strip_structured", "zhao1", "zhao2", "zhao2t3", "zhao2t3_structured"]
    check("method_schedule_identity", methods["main_methods"] == expected_methods
          and methods["ablation_methods"] == ["zhao2t3_sturm", "zhao2t3_sparse"]
          and methods["main_repetitions"] == 3 and methods["ablation_repetitions"] == 1
          and methods["grids"] == [8, 32, 128, 512] and methods["time_grids"] == [32, 64, 128, 256]
          and methods["budget_seconds"] == 5 and methods["bootstrap_seed"] == 20261012947)
    status_counts = Counter()
    threshold_by_group = {}
    for index, (case, meta) in enumerate(zip(cases, private["cases"])):
        name = case["name"]
        initial = list(map(F, case["initial"]))
        prior = list(map(F, case["prior"]))
        past = [tuple(map(F, row)) for row in case["past"]]
        future = [tuple(map(F, row)) for row in case["future"]]
        packet = [tuple(map(F, row)) for row in case["packet"]]
        horizon, rho, limit = map(F, (case["H"], case["rho"], case["limit"]))
        check(name+":physical_domain", len(initial) == len(packet) == 3 and 0 < prior[0] < prior[1]
              and rho >= 0 and horizon > 0 and all(dt > 0 for dt, _ in past+future)
              and sum(dt for dt, _ in future) == horizon and all(lo < hi for lo, hi in packet))
        hidden = F(meta["hidden_theta"])
        check(name+":hidden_prior", prior[0] < hidden < prior[1])
        terminal = [tuple(map(F, pair)) for pair in meta["certified_terminal"]]
        check(name+":stored_generation_certificate_inside_packet", len(terminal) == 3
              and all(lo <= a <= b <= hi for (lo, hi), (a, b) in zip(packet, terminal))
              and meta["history_packet_containment"] == [True, True, True])
        accel = max(abs(initial[2]), *(abs(u) for _, u in past),
                    max(abs(u) for _, u in future)+prior[1]*rho)
        bound = packet[0][1]+max(F(0), packet[1][1])*horizon+accel*horizon*horizon/2
        check(name+":public_bound_arithmetic", F(meta["public_upper"]) == bound
              and F(meta["public_acceleration_bound"]) == accel)
        rotation = meta["design"].get("threshold_rotation", 0)
        expected_rotation = 0 if index < 36 else 2 if index < 40 else 1 if index < 44 else 3
        check(name+":predeclared_rotation", rotation == expected_rotation)
        policy_index = (index+rotation) % 4
        check(name+":policy_assignment", meta["threshold_policy"] == POLICIES[policy_index])
        reference = F(meta["threshold_reference_peak"])
        margin = (1+abs(bound))/20 if policy_index == 0 else (1+abs(reference))*(F(1, 100), F(1, 10000), F(-1, 1000))[policy_index-1]
        check(name+":threshold_arithmetic", F(meta["margin"]) == margin
              and limit == (bound if policy_index == 0 else reference)+margin)
        numerical = meta["numerical_compatible_reference"]
        status_counts[numerical["status"]] += 1
        if numerical["status"] == "NUMERICAL_SEARCH_FAILED_PUBLIC_BOUND_FALLBACK":
            check(name+":retained_failure_policy", reference == bound and "error" in numerical)
        else:
            check(name+":reference_protocol", numerical["coarse"]["grid_size"] == 512
                  and numerical["fine"]["grid_size"] == 1024
                  and numerical["missing_components_or_isolated_points_not_ruled_out"] is True
                  and reference == F(numerical["reference160"]["peak"]))
            check(name+":reference_not_less_than_generated_point", reference >= F(meta["peak_at_160_dps"]["peak"]))
        threshold_by_group.setdefault(case["group"], Counter())[meta["threshold_policy"]] += 1
    math_path = ROOT/"qa/GENERATION_MATH_AUDIT.json"
    math_summary = "Not run by this low-cost audit; separate math audit pending."
    if math_path.is_file():
        separate = json.loads(math_path.read_text(encoding="utf-8"))
        math_summary = dict(separate_report="qa/GENERATION_MATH_AUDIT.json", status=separate.get("status"),
                            input_hash_matches=separate.get("input_sha256") == EXPECTED_INPUT_HASH,
                            private_hash_matches=separate.get("private_sha256") == digest(ROOT/"qa/holdout_generation_private.json"))
    report = dict(status="PASS" if not failures else "FAIL", scope="LOW_COST_METADATA_AND_EXACT_RATIONAL_ARITHMETIC_ONLY",
                  high_precision_trajectory_recomputation=math_summary,
                  expected_main_jobs=864, expected_ablation_jobs=96, checks=len(checks),
                  method_freeze_sha256=EXPECTED_METHOD_HASH, input_sha256=EXPECTED_INPUT_HASH,
                  frozen_manifest_sha256=digest(ROOT/"inputs/HOLDOUT_FROZEN.json"),
                  private_sha256=digest(ROOT/"qa/holdout_generation_private.json"),
                  group_counts=dict(counts), threshold_counts=dict(thresholds),
                  threshold_by_group={key: dict(value) for key, value in threshold_by_group.items()},
                  numerical_reference_statuses=dict(status_counts), failures=failures, details=checks,
                  limitations=["Stored trajectory certificates checked for schema and packet inclusion only; mathematical bounds not recomputed here.",
                               "Finite numerical reference is not a certified complete feasible set or global maximum.",
                               "Analyzer source frozen separately before first solve; this audit checks the method freeze supplied by coordinator."])
    output = ROOT/args.output
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
