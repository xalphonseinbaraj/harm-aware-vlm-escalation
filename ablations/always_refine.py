"""Skip the gate -- MLLM refines every case, causal verify still runs.
Answers: does the GATE add value, or would refining everything be just as
good (at much higher API cost)?
Thin wrapper around run_pipeline.py -- see that file for the actual logic.
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "run_pipeline.py", "--fusion-mode", "always_refine", *sys.argv[1:]])
