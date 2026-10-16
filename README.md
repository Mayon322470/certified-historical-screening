# Historical screening for task-oriented set-membership estimation

Research data and code accompanying **Certified Historical Screening for Task-Oriented Set-Membership Estimation**, by Yongsheng Ma, Guobao Zhang, and Yongming Huang (Southeast University). The manuscript is unpublished. Correspondence: Yongming Huang, `mayon322470@seu.edu.cn`, [ORCID 0000-0002-9697-7515](https://orcid.org/0000-0002-9697-7515).

The repository contains the original input cases, independently implemented comparison estimators, screening routines, run-level outputs, validation records, and in-house CARLA/CarSim coupling code for the three reported studies. It preserves all deadline failures, unresolved outcomes, and slower cases in the reported workloads. No experiment was rerun to prepare this release.

## Contents and experimental denominators

| Directory | Study | Retained observations |
|---|---|---|
| `research_zhao2026_comparison_20261012` | Exact lag model | 48 cases × 6 configurations × 3 repetitions = 864 principal runs; 96 component-ablation runs |
| `research_task_face_budget_20261013` | Common response error | 32 cases × 5 configurations × 3 repetitions = 480 runs |
| `vehicle_estimator_closed_loop_20261013` | Biased digital observations | 20 independently simulated trajectories, 480,020 native samples; six identical queries × five configurations × three repetitions = 90 separate timing runs |
| `vehicle_position_contract_20261013`, `vehicle_validation_20261012` | Vehicle support | Original position backup, coupling, tracker, and native solver wrapper code |
| `research_efficiency_20261012` | Independent verification support | Transitive checker source dependencies |
| `tools` | Public archive utilities | Read-only data checks, lossless restoration, and clean model-reproduction preparation |

Original directory names are retained so that relative imports keep their scientific meaning. These names are identifiers of archived studies, not publication claims. Input generation witnesses with `private` in their names are now included for offline verification. They were isolated from the evaluation solvers and must not be supplied to a solver as extra information.

## Quick verification (no simulator required)

Python 3.12 is recommended. The archived experimental environment was Python 3.12.7 on an AMD Ryzen 7 8845H with one numerical thread. From the repository root:

```console
python -B tools/verify_repository.py
```

This uses only the Python standard library. It checks every provenance-bound payload, reconstructs SHA-256 digests of compressed byte streams, reads all 864 + 96 + 480 model records, recomputes table costs and completion counts, counts all 480,020 native samples, recomputes conservative body clearance, and rebuilds complete costs from all 90 timing rows. It starts neither an estimator nor a simulator. A retained successful check is in `verification/ARCHIVED_DATA_CHECK.json`.

The quick checker verifies archived data and reported aggregates. It does not replace the original mathematical proof checkers or establish new continuous-time vehicle guarantees. Original independent checker source and reports are included alongside the studies.

## Large raw files and byte identity

Large JSONL and CSV streams are stored as deterministic gzip parts. Each part contains the next contiguous 32 MiB of the original bytes; a JSON record may span parts. `tools/repository_data.py` handles this automatically. Do not parse individual parts as standalone JSONL files.

`PROVENANCE.json` records original relative paths, byte-identical source hashes, compressed-part hashes, decoded-part hashes, and the hash of the complete reconstructed stream. `SHA256SUMS` covers the released files. Restore all original streams into a new directory with:

```console
python -B tools/restore_archives.py --output ../restored_raw_data
```

Restoration requires approximately 2.2 GB of free space and refuses an existing output directory. Compression and chunking do not remove records, digits, or proof payloads.

`.gitattributes` disables automatic line-ending conversion so that cloned files retain their recorded byte identities on Windows and Unix.

## Recompute model experiments

Install the numerical dependencies in an isolated Python environment:

```console
python -m pip install -r requirements.txt
python -B tools/prepare_model_reproduction.py --output ../fresh_model_reproduction
```

The preparation command copies frozen model source and inputs into a new workspace without the archived result directories. It does not change the original methods or input bytes. The new workspace contains both model study directories. Run the commands below there, sequentially and without competing simulation or audit jobs:

```console
python -B research_zhao2026_comparison_20261012/implementation/run_comparison.py --inputs holdout_48.json --tag reproduced_main --grids 8,32,128,512 --nt-schedule 32,64,128,256 --budget 5 --reps 3 --quick
python -B research_zhao2026_comparison_20261012/implementation/run_comparison.py --inputs holdout_48.json --tag reproduced_ablation --grids 8,32,128,512 --nt-schedule 32,64,128,256 --budget 5 --reps 1 --quick --methods zhao2t3_sturm,zhao2t3_sparse
python -B research_zhao2026_comparison_20261012/analysis/analyze_comparison.py reproduced_main
python -B research_zhao2026_comparison_20261012/analysis/analyze_comparison.py reproduced_ablation
python -B research_task_face_budget_20261013/implementation/run_budget_comparison.py
```

The 32-case statistics command, run after its timing experiment, is:

```console
python -B research_task_face_budget_20261013/analysis/analyze_comparison.py --output reproduced_statistics_32
```

Wall-clock timings and near-deadline completion can vary with hardware, Python/numerical libraries, and system load. The budget is checked at cell boundaries. Actual elapsed time determines whether a result met its deadline.

The original checker entry points are `qa/run_independent_audit.py` for the 48-case study and `audit/audit_formal_results.py` for the 32-case study. Some historical audit manifests bind absolute paths from the original machine. They are retained as provenance, not advertised as portable full-package verification. Use `tools/verify_repository.py` for the public snapshot. Restoring raw records and porting a historical independent audit requires mapping those recorded paths to the matching public files while retaining their SHA-256 identities.

The six identical vehicle queries can also be replayed without either simulator:

```console
python -B tools/replay_vehicle_queries.py --check-only
python -B tools/replay_vehicle_queries.py --output ../fresh_vehicle_query_replay
```

The first command checks source identities, imports the original estimation module, and independently selects the same six inputs without making an estimator call. The second invokes the unchanged archived replay function with its output directory redirected in memory. It creates 90 new timing runs and preserves the archived measurements. Neither command starts CARLA or CarSim. The portable wrapper is new packaging code; it does not change the historical estimator or screening source.

## Interpreting the comparisons

- **48 cases:** table cost is the sum of the per-case median times, including unresolved and late work. It is not the sum of all 144 repetitions per method.
- **32 cases:** table cost is the sum of all 96 runs per method. The pooled Zhao paired timing interval crosses one; Ma is slower on the common completed-query pairs. Common-response coupling faces exclude zero extra cells in this suite.
- **Vehicle queries:** complete cost includes preparation, estimation, and guard evaluation. The 90 query runs are not additional trajectories. The analytical range inverse is the fastest and tightest reference in this quadratic task.
- **Vehicle trajectories:** evidence consists of finite sampled CARLA/CarSim trajectories with ideal digitally corrupted observations. CarSim computes vehicle forces; the CARLA proxy supplies pose and task geometry with its physics disabled. This is not a physical vehicle experiment. Full-cycle wall times are retained; query deadlines do not establish hard real-time execution of the entire simulation loop.

The Ma and Zhao routines are independent adaptations to the stated history-consistency queries. They are not the original authors' distributed software or a reproduction of every application in those papers. Their references are:

1. Y. Ma, Z. Wang, M. Kinnaert, and Y. Shen, “Guaranteed state estimation for nonlinear systems using a less conservative zonotopic first-order expansion method,” *Automatica* 192 (2026), 113189. https://doi.org/10.1016/j.automatica.2026.113189
2. Z. Zhao, Z. Wang, J. Liang, and W. Xu, “Zonotopic Polynomial Set-Membership Fusion Estimation for Nonlinear Systems via Carleman Approximation: An Encoding–Decoding Scheme,” *IEEE Transactions on Automatic Control* 71(4) (2026), 2466–2481. https://doi.org/10.1109/TAC.2025.3627100

## Vehicle co-simulation requirements

Offline inspection and archived-data verification require no CARLA or CarSim installation. Regenerating the 20 trajectories requires Windows, CARLA **0.10.0** with **Town10HD_Opt**, a matching Python API, CarSim **2024.1 Revision 222209**, a valid local CarSim license, and the separately installed `SDK_External_Control.cpar` vehicle archive. The commercial archive, CarSim solver DLL, and derived solver parameter/binary files are not distributed here.

The in-house sources are provided in `vehicle_estimator_closed_loop_20261013/integration/run_estimator_closed_loop.py`, `vehicle_position_contract_20261013/integration/coupling_session_v3.py`, and `vehicle_validation_20261012/source/cg_vehicle.py`. They retain their original source bytes and frozen provenance. Original absolute installation paths and hardware gates require adaptation to the user's local simulator installation. `external_simulator/validation_config.example.json` lists the public settings and placeholders. Original archived freeze hashes must not be represented as hashes of modified source. After adapting a new working copy, regenerate its component checks and protocol freeze before a new simulation study; do not overwrite or relabel the retained 20 trajectories.

The simulation uses ports 2100–2102 and an exclusive simulation lock. The bridge starts and cleans up only its own server. Licensed simulator execution is not required by any command in the quick-verification section.

## Attribution, availability, and rights

See `CITATION.cff` for the unpublished accompanying work. A publication DOI can be added after publication. The repository does not include a repository-wide reuse license; public access alone does not grant an MIT or other software license. Third-party commercial software and journal full texts are excluded.
