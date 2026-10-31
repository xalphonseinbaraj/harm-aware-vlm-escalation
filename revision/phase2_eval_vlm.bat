@echo off
REM ===================================================================================
REM PHASE 2 -- one VLM query per test image for every (backbone, VLM) pair.
REM All escalation policies are then computed offline by analyze.py from these logs.
REM eval_all.py RESUMES automatically if interrupted -- just run the same line again.
REM
REM Before starting:
REM   ollama pull qwen2.5vl:7b
REM   (Llama-3.2-11B) NIM key in .env as NVIDIA_NIM_API_KEY.  If NIM no longer works:
REM        ollama pull llama3.2-vision:11b   and set LLAMA=llama11b_p1_ollama below.
REM Dry run first (no VLM, ~2 min):
REM   python revision\eval_all.py --backbone ckpt_rev\strong_s0\best.pt --gate ckpt_rev\strong_s0\gate.joblib --vlm mock --out results_rev\_dryrun__mock.jsonl
REM   (delete results_rev\_dryrun__mock.jsonl afterwards)
REM ===================================================================================
cd /d E:\polyp_llm_seg
set LLAMA=llama11b_p1_nim
set QWEN=qwen7b_p3

for %%B in (strong_s0 strong_s1 strong_s2 light_s0 light_s1 light_s2) do (
    python revision\eval_all.py --backbone ckpt_rev\%%B\best.pt --gate ckpt_rev\%%B\gate.joblib --vlm %QWEN% --out results_rev\%%B__%QWEN%.jsonl --save-masks masks_rev\%%B__%QWEN% || goto :fail
    python revision\eval_all.py --backbone ckpt_rev\%%B\best.pt --gate ckpt_rev\%%B\gate.joblib --vlm %LLAMA% --workers 6 --out results_rev\%%B__%LLAMA%.jsonl --save-masks masks_rev\%%B__%LLAMA% || goto :fail
)
REM control that separates the MODEL from the PROMPT (Qwen with the plain prompt P1), seed 0 only
python revision\eval_all.py --backbone ckpt_rev\strong_s0\best.pt --gate ckpt_rev\strong_s0\gate.joblib --vlm qwen7b_p1 --out results_rev\strong_s0__qwen7b_p1.jsonl || goto :fail

REM lambda sensitivity: backbone-only passes (no VLM needed)
for %%B in (strong_s0_lam0.1 strong_s0_lam1.0) do (
    python revision\eval_all.py --backbone ckpt_rev\%%B\best.pt --gate ckpt_rev\%%B\gate.joblib --out results_rev\lambda_%%B.jsonl || goto :fail
)
echo PHASE 2 DONE
goto :eof
:fail
echo *** PHASE 2 FAILED -- re-run the same command; it resumes where it stopped ***
exit /b 1
