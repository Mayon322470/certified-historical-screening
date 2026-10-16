"""Independent packet-envelope baseline; writes only this analysis directory.

Run only after the main benchmark finishes and the parent authorizes timing:
  python quick_baseline.py --run

The timed region includes input parsing, the exact packet bound, and one
historical midpoint endpoint on EVERY trial. No future propagation is used.
The shared 96-bit interval endpoint core is an explicit trust boundary.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "implementation"))
import certified_core as cc


def encode(obj):
    if isinstance(obj, F):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): encode(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [encode(v) for v in obj]
    return obj


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quick_certificate(raw_case):
    """Certify all packet states, plus a strict nonempty midpoint witness."""
    initial = tuple(map(F, raw_case["initial"]))
    prior = tuple(map(F, raw_case["prior"]))
    past = [tuple(map(F, row)) for row in raw_case["past"]]
    future = [tuple(map(F, row)) for row in raw_case["future"]]
    packet = [tuple(map(F, row)) for row in raw_case["packet"]]
    H, rho, limit = (F(raw_case[key]) for key in ("H", "rho", "limit"))
    if not (0 < prior[0] < prior[1] and H >= 0 and rho >= 0):
        raise ValueError("Invalid prior, horizon, or disturbance bound")
    if len(packet) != 3 or any(lo > hi for lo, hi in packet):
        raise ValueError("Invalid coordinate packet")
    if any(dt < 0 for dt, _ in past + future):
        raise ValueError("Negative hold duration")
    if sum((dt for dt, _ in future), F(0)) < H:
        raise ValueError("Future command record does not cover H")

    U = max((abs(u) for _, u in future), default=F(0))
    A_packet = max(abs(packet[2][0]), abs(packet[2][1]))
    A_bar = max(A_packet, U) + rho * prior[1]
    upper = packet[0][1] + max(packet[1][1], F(0)) * H + A_bar * H * H / 2
    bound_pass = upper <= limit

    # Deliberately evaluate the midpoint on every timed trial, even if the
    # bound already fails. This makes the measured quick-gate cost explicit.
    midpoint = (prior[0] + prior[1]) / 2
    endpoint = cc.endpoint(midpoint, initial, past)
    margins = [
        (endpoint[i].lower() - packet[i][0], packet[i][1] - endpoint[i].upper())
        for i in range(3)
    ]
    midpoint_inside = all(left > 0 and right > 0 for left, right in margins)
    safe = bound_pass and midpoint_inside
    reasons = []
    if not bound_pass:
        reasons.append("PACKET_ENVELOPE_EXCEEDS_LIMIT")
    if not midpoint_inside:
        reasons.append("STRICT_MIDPOINT_NONEMPTY_WITNESS_NOT_CERTIFIED")

    return dict(
        status="QUICK_CERTIFIED_SAFE" if safe else "QUICK_NO_CERTIFICATE",
        bound_proves_universal_safety=bound_pass,
        midpoint_strictly_inside=midpoint_inside,
        unresolved_reasons=reasons,
        A_packet=A_packet,
        U=U,
        A_bar=A_bar,
        H=H,
        rho=rho,
        limit=limit,
        quick_upper=upper,
        safety_margin=limit - upper,
        midpoint_theta=midpoint,
        midpoint_state_bounds=[(iv.lower(), iv.upper()) for iv in endpoint],
        midpoint_state_fixedpoint=[iv.serialized() for iv in endpoint],
        midpoint_packet_margins=margins,
    )


def write_report(records, cases, tasks, summary, provenance):
    primary = [r for r in records if r["nt"] == 64]
    by_case = defaultdict(list)
    for row in primary:
        by_case[row["case_index"]].append(row)
    group_counts = defaultdict(Counter)
    for i, case in enumerate(cases):
        result = by_case[i][0]["result"]
        group_counts[case["group"]]["cases"] += 1
        group_counts[case["group"]]["bound_pass"] += int(result["bound_proves_universal_safety"])
        group_counts[case["group"]]["midpoint_pass"] += int(result["midpoint_strictly_inside"])
        group_counts[case["group"]]["quick_safe"] += int(result["status"] == "QUICK_CERTIFIED_SAFE")
    lines = [
        "第一篇：廉价测量包安全基线核查", "日期：2026-10-12", "",
        "范围：冻结43例的主任务及同一清单指定的10个nt=256任务，共53任务；每任务3次冷指数缓存，共159记录。",
        "没有修改任何主实验代码、冻结输入或954条固定网格结果。主实验完成后才运行本分析。", "",
        "解析界：A_bar=max(max(|Ba_lower|,|Ba_upper|),U)+rho*theta_hi；",
        "x(s)<=Bx_upper+max(Bv_upper,0)*H+A_bar*H^2/2，覆盖0<=s<=H。",
        "仅当此精确有理上界<=L，且一次prior中点历史endpoint的整个外包严格位于packet内部，才返回QUICK_CERTIFIED_SAFE。",
        "每次计时均包含输入有理化、解析界与这一次历史endpoint，即使解析界已失败也计算中点；不做任何未来timeline或mesh。", "",
        f"43个独立输入：包界已证明全体包状态安全 {summary['unique_cases_bound_pass']} 例；中点严格非空见证通过 {summary['unique_cases_midpoint_pass']} 例；两者同时通过 {summary['unique_cases_quick_safe']} 例。",
        f"53个计时任务：quick SAFE {summary['task_quick_safe']} 项；其余 {len(tasks)-summary['task_quick_safe']} 项仅返回NO_CERTIFICATE。",
        f"159条记录全部future timeline调用为0，三次科学结果一致={summary['repeat_scientific_results_identical']}。", "",
        "分组计数（43个独立输入，不重复计算长时间网格任务）：",
        "group | cases | 包界通过 | 中点非空通过 | quick SAFE",
    ]
    for group, counts in group_counts.items():
        lines.append(f"{group} | {counts['cases']} | {counts['bound_pass']} | {counts['midpoint_pass']} | {counts['quick_safe']}")
    lines += ["", "主任务逐例结果：", "case | 包界通过 | 中点通过 | quick状态 | 三次中位时间(ms)"]
    for i, case in enumerate(cases):
        rows = by_case[i]
        result = rows[0]["result"]
        median_ms = statistics.median(row["elapsed_seconds"] for row in rows) * 1000
        lines.append(f"{case['name']} | {result['bound_proves_universal_safety']} | {result['midpoint_strictly_inside']} | {result['status']} | {median_ms:.6f}")
    lines += [
        "", "如何解读宽裕案例：",
        "varied_rational的limit生成式刻意宽裕。因此凡包界通过者，都已由不含未来网格的解析式证明了全体真实可行状态的安全。",
        "若中点见证失败，只表示这个单点未认证可行集非空；既不否定解析安全上界，也不证明实例不安全或不一致。",
        "该组packet生成中心用theta=3/2，prior=[1/2,3]的中点却是7/4，所以尤其窄包可能出现上述区别。没有为本基线按结果改选见证点。", "",
        "主文应诚实纳入：",
        "1. 固定网格工作量数据可作为收缩/延迟传播消融，但宽裕案例不能被用来主张必须采用复杂新方法才能快速判定安全。",
        "2. 该quick gate可统一置于所有对比方法之前；门通过即返回SAFE。若仅包界通过，是否接受条件式/可能空集安全须按统一输出语义处理。",
        "3. 要评价真正的端到端优势，应计入共同quick gate，并比较门未解决的任务；原954记录保留，不冒充加入gate后的重跑。",
        "4. 此门很弱且有意便宜；门未通过不是新方法胜出。需要后续更强通用方法才能处理紧约束、相关性或中点见证缺失。", "",
        "证据和计时边界：",
        "解析上界与不等号比较使用Fraction精确算术。非空见证复用当前96-bit certified_core.endpoint，并明确保留这一算术核心依赖。",
        "结果按重复完全一致检查；完整每次计时、工作量、包界、中点外界及所用指数记录保存在JSON。",
        "冷缓存指每次清空指数lru_cache及指数证书字典，不表示重启Python/操作系统或测得WCET。计时不含导入、清缓存和JSON序列化。",
        "本分析与主实验不并发；未消除机器其余进程/热状态噪声。未直接将这些时刻与此前主实验时刻宣称为公平速度比。",
        f"输入SHA-256：{provenance['input_sha256']}",
        f"核心SHA-256：{provenance['core_sha256']}",
        f"脚本SHA-256：{provenance['script_sha256']}", "",
    ]
    (HERE / "QUICK_BASELINE_REPORT_CN.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run only after the parent announces main timing has finished")
    args = parser.parse_args()
    if not args.run:
        print("Prepared only. No timing executed. Use --run after main benchmark completion and parent authorization.")
        return

    outputs = [HERE / name for name in (
        "quick_baseline_results.json", "quick_baseline_exp_certificates.json", "QUICK_BASELINE_REPORT_CN.txt"
    )]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Analysis outputs already exist; preserve them")
    main_path = ROOT / "implementation/benchmark_results.json"
    if not main_path.is_file():
        raise RuntimeError("Main benchmark final output absent; do not overlap timing")
    main_records = json.loads(main_path.read_text(encoding="utf-8"))
    if len(main_records["records"]) != 954:
        raise RuntimeError("Expected 954 completed main records before timing this baseline")
    del main_records

    raw = (ROOT / "inputs/fixtures.json").read_bytes()
    frozen = json.loads((ROOT / "inputs/FROZEN_INPUTS.json").read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest() != frozen["sha256"]:
        raise ValueError("Frozen input hash mismatch")
    cases = json.loads(raw)
    if len(cases) != 43 or len(frozen["long_workload_case_indices"]) != 10:
        raise ValueError("Frozen task inventory changed")
    tasks = [(i, 32, 64) for i in range(len(cases))]
    tasks += [(i, 32, 256) for i in frozen["long_workload_case_indices"]]
    schedule = [(rep, i, n, nt) for rep in range(3) for i, n, nt in tasks]
    random.Random(2026101202).shuffle(schedule)
    provenance = dict(
        input_sha256=hashlib.sha256(raw).hexdigest(),
        frozen_metadata_sha256=digest(ROOT / "inputs/FROZEN_INPUTS.json"),
        core_sha256=digest(ROOT / "implementation/certified_core.py"),
        script_sha256=digest(Path(__file__)),
        completed_main_sha256=digest(main_path),
        python=sys.version,
        platform=platform.platform(),
        clock=time.get_clock_info("perf_counter")._asdict() if hasattr(time.get_clock_info("perf_counter"), "_asdict") else vars(time.get_clock_info("perf_counter")),
    )
    records, exponents, seen = [], {}, {}
    counts = Counter()
    original_flow, original_endpoint, original_timeline = cc.flow, cc.endpoint, cc.timeline

    def counted_flow(*args, **kwargs):
        counts["history_flow_calls"] += 1
        return original_flow(*args, **kwargs)

    def counted_endpoint(*args, **kwargs):
        counts["historical_endpoint_calls"] += 1
        return original_endpoint(*args, **kwargs)

    def forbidden_timeline(*args, **kwargs):
        counts["future_timeline_calls"] += 1
        raise AssertionError("The packet-envelope baseline must not propagate the future")

    cc.flow, cc.endpoint, cc.timeline = counted_flow, counted_endpoint, forbidden_timeline
    try:
        for sequence, (rep, index, n, nt) in enumerate(schedule):
            cc.exp_neg.cache_clear()
            cc.EXP_CERTIFICATES.clear()
            counts.clear()
            start = time.perf_counter()
            result = quick_certificate(cases[index])
            elapsed = time.perf_counter() - start
            cache = cc.exp_neg.cache_info()
            if counts["historical_endpoint_calls"] != 1 or counts["future_timeline_calls"] != 0:
                raise AssertionError("Incorrect quick-baseline call counts")
            canonical = json.dumps(encode(result), sort_keys=True)
            if index in seen and seen[index] != canonical:
                raise ValueError("Repeated/grid-tagged scientific result differs")
            seen[index] = canonical
            for exponent, enclosure in cc.EXP_CERTIFICATES.items():
                if exponent in exponents and exponents[exponent] != enclosure:
                    raise ValueError("Repeated exponential enclosure differs")
                exponents[exponent] = enclosure
            records.append(dict(
                sequence=sequence, rep=rep, case=cases[index]["name"],
                case_index=index, group=cases[index]["group"], n=n, nt=nt,
                grid_tags_only=True, elapsed_seconds=elapsed,
                work=dict(counts, future_timeline_calls=counts["future_timeline_calls"],
                          exponential_evaluations=cache.misses,
                          exponential_cache_hits=cache.hits),
                exponential_arguments=list(cc.EXP_CERTIFICATES), result=result,
            ))
    finally:
        cc.flow, cc.endpoint, cc.timeline = original_flow, original_endpoint, original_timeline

    unique = {row["case_index"]: row["result"] for row in records}
    task_results = {(row["case_index"], row["n"], row["nt"]): row["result"] for row in records}
    summary = dict(
        cases=len(cases), tasks=len(tasks), records=len(records),
        unique_cases_bound_pass=sum(r["bound_proves_universal_safety"] for r in unique.values()),
        unique_cases_midpoint_pass=sum(r["midpoint_strictly_inside"] for r in unique.values()),
        unique_cases_quick_safe=sum(r["status"] == "QUICK_CERTIFIED_SAFE" for r in unique.values()),
        task_quick_safe=sum(r["status"] == "QUICK_CERTIFIED_SAFE" for r in task_results.values()),
        record_status_counts=dict(Counter(row["result"]["status"] for row in records)),
        unique_exponential_enclosures=len(exponents),
        repeat_scientific_results_identical=True,
        future_timeline_calls=sum(row["work"]["future_timeline_calls"] for row in records),
        timing_median_seconds=statistics.median(row["elapsed_seconds"] for row in records),
        timing_max_seconds=max(row["elapsed_seconds"] for row in records),
    )
    result = dict(
        status="COMPLETED", provenance=provenance, bits=cc.BITS,
        schedule_seed=2026101202, summary=summary,
        scientific_scope="Exact packet-envelope comparison plus strict midpoint nonempty witness, sharing the historical interval endpoint core; no future propagation.",
        timing_scope="Per-call cold exponential cache; parse, packet bound and one midpoint endpoint included in every trial. Imports, cache clearing, output serialization excluded. Not WCET or a directly paired runtime speedup experiment.",
        records=records,
    )
    outputs[0].write_text(json.dumps(encode(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs[1].write_text(json.dumps(dict(bits=cc.BITS, intervals=exponents), indent=2) + "\n", encoding="utf-8")
    write_report(records, cases, tasks, summary, provenance)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
