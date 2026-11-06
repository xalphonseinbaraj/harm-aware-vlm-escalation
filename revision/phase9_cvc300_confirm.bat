@echo off
REM ===================================================================================
REM PHASE 9 -- confirmation of the learned acceptance rule on an UNUSED external dataset (CVC-300, 60 images)
REM   The acceptance rule is fixed: fitted on the Kvasir-SEG validation split only (revision\learned_acceptance.py),
REM   unchanged since the v13 manuscript. CVC-300 has not been used anywhere in the study.
REM   Works OFFLINE: MedSAM is already in the Hugging Face cache; Ollama is NOT needed.
REM   MedSAM runs on the CPU (the GPU driver hangs with MedSAM). About 1 hour. Resumes if interrupted.
REM ===================================================================================
cd /d E:\polyp_llm_seg
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set MEDSAM_DEVICE=cpu
set D=data\TestDataset\CVC-300
for %%B in (strong_s0 strong_s1 strong_s2) do (
    python revision\eval_all.py --backbone ckpt_rev\%%B\best.pt --gate ckpt_rev\%%B\gate.joblib --vlm medsam_box --data-root %D% --out results_ext\CVC-300_%%B__medsam_box.jsonl || goto :fail
)
python revision\eval_all.py --backbone ckpt_rev\base_segformer_mitb2\best.pt --gate ckpt_rev\base_segformer_mitb2\gate.joblib --vlm medsam_box --data-root %D% --out results_ext\CVC-300_segformer_s0__medsam_box.jsonl || goto :fail
for %%S in (1 2) do (
    python revision\eval_all.py --backbone ckpt_rev\base_segformer_mitb2_s%%S\best.pt --gate ckpt_rev\base_segformer_mitb2_s%%S\gate.joblib --vlm medsam_box --data-root %D% --out results_ext\CVC-300_segformer_s%%S__medsam_box.jsonl || goto :fail
)
python revision\learned_acceptance.py || goto :fail
echo.
echo PHASE 9 DONE
goto :eof
:fail
echo *** PHASE 9 STOPPED -- run this file again (finished images are kept). If it says it cannot find the model offline, delete the two lines "set HF_HUB_OFFLINE=1" and "set TRANSFORMERS_OFFLINE=1" and connect to the internet. ***
exit /b 1
