@echo off
REM ===================================================================================
REM PHASE 6 -- strong (ImageNet-pretrained) backbone + gate + MedGemma, in-domain and cross-dataset
REM   Why: our compact PolypUNet generalises poorly to other hospitals (IoU 0.34-0.54), so the
REM   external escalation effect is small. SegFormer-B2 generalises well (IoU 0.67-0.77), so this
REM   shows the gate protecting a strong, deployable backbone under domain shift.
REM   Gate is fitted on Kvasir only (Phase 3) -- no re-training, no re-fitting on external data.
REM About 1 to 1.5 hours. Resumes if interrupted -- just run this file again.
REM Needs Window A (ollama serve) running.
REM ===================================================================================
cd /d E:\polyp_llm_seg
set TD=data\TestDataset
set SB=ckpt_rev\base_segformer_mitb2

REM ---------- Kvasir test split
python revision\eval_all.py --backbone %SB%\best.pt --gate %SB%\gate.joblib --vlm medgemma4b_p1 --out results_rev\segformer_s0__medgemma4b_p1.jsonl || goto :fail

REM ---------- external sets
for %%D in (CVC-ColonDB ETIS-LaribPolypDB CVC-ClinicDB) do (
    python revision\eval_all.py --backbone %SB%\best.pt --gate %SB%\gate.joblib --vlm medgemma4b_p1 --data-root %TD%\%%D --out results_ext\%%D_segformer_s0__medgemma4b_p1.jsonl || goto :fail
)

REM ---------- analysis
python revision\analyze.py --dir results_rev || goto :fail
python revision\analyze.py --dir results_ext || goto :fail
echo.
echo PHASE 6 DONE
goto :eof
:fail
echo *** PHASE 6 STOPPED -- fix the message above and run this file again (finished images are kept) ***
exit /b 1
