@echo off
REM PHASE 4 -- all tables + figures (a few minutes, no GPU needed)
cd /d E:\polyp_llm_seg
python revision\analyze.py --dir results_rev --theta 0.5 --tau 0.5 || goto :fail
python revision\make_qual_figure.py --ours results_rev\strong_s0__qwen7b_p3.jsonl --ours-masks masks_rev\strong_s0__qwen7b_p3 --out results_rev\analysis\figures\fig_qualitative_qwen.png || goto :fail
python revision\make_qual_figure.py --ours results_rev\strong_s0__llama11b_p1_nim.jsonl --ours-masks masks_rev\strong_s0__llama11b_p1_nim --out results_rev\analysis\figures\fig_qualitative_llama.png || goto :fail
echo.
echo PHASE 4 DONE -- results are in results_rev\analysis (tables.md, summary.json, figures\)
echo plus ckpt_rev\*\summary.json and ckpt_rev\*\gate.val_report.json
goto :eof
:fail
echo *** PHASE 4 FAILED ***
exit /b 1
