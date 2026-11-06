@echo off
REM ===================================================================================
REM PHASE 7 -- two strengthening experiments (about 8-10 hours in total; run overnight)
REM   A. VLM answers on the VALIDATION split, so that
REM        (i)  the pre-deployment check is computed on validation and verified on the untouched test split, and
REM        (ii) a gate trained on edit benefit can be fitted on validation and evaluated on test (no test-set cross-fitting).
REM   B. Two more SegFormer training seeds (1 and 2), so that the SegFormer results are no longer single-run.
REM Every step resumes if interrupted -- just run this file again.
REM Needs Window A: set OLLAMA_MODELS=E:\ollama_models  then  ollama serve
REM ===================================================================================
cd /d E:\polyp_llm_seg
set TD=data\TestDataset

REM ---------- A: validation-split VLM answers (strong PolypUNet seeds 0-2, SegFormer seed 0)
for %%B in (strong_s0 strong_s1 strong_s2) do (
    python revision\eval_all.py --split val --backbone ckpt_rev\%%B\best.pt --gate ckpt_rev\%%B\gate.joblib --vlm medgemma4b_p1 --out results_val\%%B__medgemma4b_p1.jsonl || goto :fail
    python revision\eval_all.py --split val --backbone ckpt_rev\%%B\best.pt --gate ckpt_rev\%%B\gate.joblib --vlm qwen7b_p3 --out results_val\%%B__qwen7b_p3.jsonl || goto :fail
)
python revision\eval_all.py --split val --backbone ckpt_rev\base_segformer_mitb2\best.pt --gate ckpt_rev\base_segformer_mitb2\gate.joblib --vlm medgemma4b_p1 --out results_val\segformer_s0__medgemma4b_p1.jsonl || goto :fail

REM ---------- B: SegFormer seeds 1 and 2 (train, fit gate, test, validation, external)
for %%S in (1 2) do (
    if not exist ckpt_rev\base_segformer_mitb2_s%%S\best.pt python revision\train_baselines.py --model segformer_mitb2 --seed %%S --batch 4 --out ckpt_rev\base_segformer_mitb2_s%%S || goto :fail
    if not exist ckpt_rev\base_segformer_mitb2_s%%S\gate.joblib python revision\fit_gate.py --backbone ckpt_rev\base_segformer_mitb2_s%%S\best.pt --out ckpt_rev\base_segformer_mitb2_s%%S\gate.joblib || goto :fail
    python revision\eval_all.py --backbone ckpt_rev\base_segformer_mitb2_s%%S\best.pt --gate ckpt_rev\base_segformer_mitb2_s%%S\gate.joblib --vlm medgemma4b_p1 --out results_rev\segformer_s%%S__medgemma4b_p1.jsonl || goto :fail
    python revision\eval_all.py --backbone ckpt_rev\base_segformer_mitb2_s%%S\best.pt --gate ckpt_rev\base_segformer_mitb2_s%%S\gate.joblib --vlm qwen7b_p3 --out results_rev\segformer_s%%S__qwen7b_p3.jsonl || goto :fail
    python revision\eval_all.py --split val --backbone ckpt_rev\base_segformer_mitb2_s%%S\best.pt --gate ckpt_rev\base_segformer_mitb2_s%%S\gate.joblib --vlm medgemma4b_p1 --out results_val\segformer_s%%S__medgemma4b_p1.jsonl || goto :fail
    for %%D in (CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB) do (
        python revision\eval_all.py --backbone ckpt_rev\base_segformer_mitb2_s%%S\best.pt --gate ckpt_rev\base_segformer_mitb2_s%%S\gate.joblib --vlm medgemma4b_p1 --data-root %TD%\%%D --out results_ext\%%D_segformer_s%%S__medgemma4b_p1.jsonl || goto :fail
    )
)

REM ---------- analysis
python revision\analyze.py --dir results_rev || goto :fail
python revision\analyze.py --dir results_ext || goto :fail
python revision\analyze.py --dir results_val || goto :fail
echo.
echo PHASE 7 DONE
goto :eof
:fail
echo *** PHASE 7 STOPPED -- fix the message above and run this file again (finished images are kept) ***
exit /b 1
