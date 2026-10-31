"""Gate + MLLM refine gated-in cases, but accept every refinement blindly --
no causal verification step.
Answers: does VERIFICATION add value, or is the MLLM trustworthy enough to
accept outright? This is the ablation that most directly targets the paper's
"trust but verify" claim.
Thin wrapper around run_pipeline.py -- see that file for the actual logic.
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "run_pipeline.py", "--fusion-mode", "no_verify", *sys.argv[1:]])
