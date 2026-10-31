"""Floor baseline: backbone mask only, no gate/MLLM/verification at all.
Answers: does the refinement pipeline help AT ALL over the plain backbone?
Thin wrapper around run_pipeline.py -- see that file for the actual logic.
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "run_pipeline.py", "--fusion-mode", "never_refine", *sys.argv[1:]])
