@echo off
REM ===================================================================================
REM PHASE 8 -- Experiment C: MedSAM as a capable escalation target (run AFTER Phase 7)
REM   MedSAM re-segments each escalated image from a box around the backbone mask, instead of a VLM choosing
REM   expand/shrink/smooth. Same gate, same verification, same policies. Tests whether the pre-deployment check
REM   says "escalate" when the target can actually improve masks.
REM   Does NOT need Ollama. First run downloads MedSAM (about 360 MB) from Hugging Face.
REM   About 1-2 hours. Resumes if interrupted -- just run this file again.
REM ===================================================================================
cd /d E:\polyp_llm_seg
set TD=data\TestDataset
uv pip install transformers || pip install transformers || goto :fail

REM ---------- Kvasir test and validation: strong PolypUNet seeds 0-2
for %%B in (strong_s0 strong_s1 strong_s2) do (
    python revision\eval_all.py --backbone ckpt_rev\%%B\best.pt --gate ckpt_rev\%%B\gate.joblib --vlm medsam_box --out results_rev\%%B__medsam_box.jsonl --save-masks masks_rev\%%B__medsam_box || goto :fail
    python revision\eval_all.py --split val --backbone ckpt_rev\%%B\best.pt --gate ckpt_rev\%%B\gate.joblib --vlm medsam_box --out results_val\%%B__medsam_box.jsonl || goto :fail
)

REM ---------- SegFormer seeds 0-2 (seeds 1-2 exist after Phase 7)
call :segformer 0 || goto :fail
call :segformer 1 || goto :fail
call :segformer 2 || goto :fail

REM ---------- external sets: strong PolypUNet seeds 0-2
for %%D in (CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB) do (
    for %%B in (strong_s0 strong_s1 strong_s2) do (
        python revision\eval_all.py --backbone ckpt_rev\%%B\best.pt --gate ckpt_rev\%%B\gate.joblib --vlm medsam_box --data-root %TD%\%%D --out results_ext\%%D_%%B__medsam_box.jsonl || goto :fail
    )
)

python revision\analyze.py --dir results_rev || goto :fail
python revision\analyze.py --dir results_ext || goto :fail
python revision\analyze.py --dir results_val || goto :fail
echo.
echo PHASE 8 DONE
goto :eof

:segformer
if "%1"=="0" (set SF=ckpt_rev\base_segformer_mitb2) else (set SF=ckpt_rev\base_segformer_mitb2_s%1)
if not exist %SF%\gate.joblib (echo skip SegFormer seed %1 - run Phase 7 first & exit /b 0)
python revision\eval_all.py --backbone %SF%\best.pt --gate %SF%\gate.joblib --vlm medsam_box --out results_rev\segformer_s%1__medsam_box.jsonl || exit /b 1
python revision\eval_all.py --split val --backbone %SF%\best.pt --gate %SF%\gate.joblib --vlm medsam_box --out results_val\segformer_s%1__medsam_box.jsonl || exit /b 1
for %%D in (CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB) do (
    python revision\eval_all.py --backbone %SF%\best.pt --gate %SF%\gate.joblib --vlm medsam_box --data-root %TD%\%%D --out results_ext\%%D_segformer_s%1__medsam_box.jsonl || exit /b 1
)
exit /b 0

:fail
echo *** PHASE 8 STOPPED -- fix the message above and run this file again (finished images are kept) ***
exit /b 1
