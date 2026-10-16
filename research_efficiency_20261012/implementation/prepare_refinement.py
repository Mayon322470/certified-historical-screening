"""Create isolated runner variants; never edit the frozen first implementation."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    source=(ROOT/'efficient_core.py').read_text()
    source=source.replace('import sparse_faces as sf','import taylor_faces as sf')
    source=source.replace("METHODS = ('eager_chord','lazy_chord','interval_faces','curvature_faces','sturm_faces','hybrid_faces')",
                          "METHODS = ('taylor_faces','taylor_curvature','hybrid_taylor')")
    source=source.replace("('interval_faces','curvature_faces','hybrid_faces')",
                          "('taylor_faces','taylor_curvature','hybrid_taylor')")
    source=source.replace("('sturm_faces','hybrid_faces')","('hybrid_taylor',)")
    source=source.replace("method!='interval_faces'","method!='taylor_faces'")
    runner=(ROOT/'run_benchmark.py').read_text().replace('from efficient_core import','from refined_core import')
    runner=runner.replace('benchmark_results.json','refined_results.json')
    runner=runner.replace('benchmark_exp_certificates.json','refined_exp_certificates.json')
    runner=runner.replace('2026101201','2026101202')
    for name,content in (('refined_core.py',source),('run_refined.py',runner)):
        if (ROOT/name).exists():
            raise FileExistsError(name)
        (ROOT/name).write_text(content,encoding='utf-8')
    print('Isolated refined implementations created.')

if __name__=='__main__':main()
