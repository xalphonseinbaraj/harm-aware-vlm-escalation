@echo off
REM ===================================================================================
REM PHASE 10 -- retrospective external evaluation on PolypGen (Ali et al., Sci Data 2023), centres C1-C6, 1,411 images
REM   Internal analysis plan in revision\PHASE10_PREREGISTRATION.txt, written before the download. Duplicate check: no PolypGen
REM   image is a duplicate of Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB or ETIS (7 closest pairs checked visually).
REM   Frozen method: MedSAM box prompt, Kvasir-SEG gate, learned acceptance rule fitted on validation only.
REM   OFFLINE, CPU only (MedSAM). PolypUNet first (primary, ~5 h), then SegFormer (secondary, ~5 h). Resumes if stopped.
REM ===================================================================================
cd /d E:\polyp_llm_seg
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set MEDSAM_DEVICE=cpu
set PG=data\PolypGen
REM ---------- primary: strong PolypUNet, three seeds
for %%B in (strong_s0 strong_s1 strong_s2) do (
    for %%C in (C1 C2 C3 C4 C5 C6) do (
        python revision\eval_all.py --backbone ckpt_rev\%%B\best.pt --gate ckpt_rev\%%B\gate.joblib --vlm medsam_box --data-root %PG%\%%C --out results_ext\PolypGen-%%C_%%B__medsam_box.jsonl || goto :fail
    )
)
python revision\polypgen_confirm.py
REM ---------- secondary: SegFormer, three seeds
for %%S in (0 1 2) do (
    for %%C in (C1 C2 C3 C4 C5 C6) do (
        if "%%S"=="0" (
            python revision\eval_all.py --backbone ckpt_rev\base_segformer_mitb2\best.pt --gate ckpt_rev\base_segformer_mitb2\gate.joblib --vlm medsam_box --data-root %PG%\%%C --out results_ext\PolypGen-%%C_segformer_s0__medsam_box.jsonl || goto :fail
        ) else (
            python revision\eval_all.py --backbone ckpt_rev\base_segformer_mitb2_s%%S\best.pt --gate ckpt_rev\base_segformer_mitb2_s%%S\gate.joblib --vlm medsam_box --data-root %PG%\%%C --out results_ext\PolypGen-%%C_segformer_s%%S__medsam_box.jsonl || goto :fail
        )
    )
)
python revision\polypgen_confirm.py || goto :fail
echo.
echo PHASE 10 DONE
goto :eof
:fail
echo *** PHASE 10 STOPPED -- run this file again (finished images are kept) ***
exit /b 1
