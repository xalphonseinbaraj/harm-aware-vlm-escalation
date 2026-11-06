"""Copy the per-image logs shipped in results/ to the folder names the analysis scripts expect. Run once from the repository root:
    python revision/setup_results_dirs.py
results/kvasir -> results_rev, results/validation -> results_val, results/external -> results_ext"""
import shutil
from pathlib import Path

for src, dst in [("results/kvasir", "results_rev"), ("results/validation", "results_val"), ("results/external", "results_ext")]:
    s, d = Path(src), Path(dst)
    if not s.exists():
        print(f"skip: {src} not found"); continue
    d.mkdir(exist_ok=True); n = 0
    for f in s.iterdir():
        if f.is_file() and not (d / f.name).exists():
            shutil.copy2(f, d / f.name); n += 1
    print(f"{src} -> {dst}: {n} files copied")
