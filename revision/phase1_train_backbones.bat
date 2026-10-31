@echo off
REM ===================================================================================
REM PHASE 1 -- backbone training for the revision  (approx. 50 min per run on RTX 4050)
REM Run from E:\polyp_llm_seg with the polyp_llm_seg venv activated:
REM     cd /d E:\polyp_llm_seg
REM     .venv\Scripts\activate
REM     revision\phase1_train_backbones.bat
REM Each run is independent; if one stops, just re-run that single line.
REM ===================================================================================
cd /d E:\polyp_llm_seg

REM --- R1 c5: stronger augmentation, 3 training seeds (split fixed: 700/150/150, seed 42)
python revision\train_backbone_v2.py --seed 0 --aug strong --out ckpt_rev\strong_s0 || goto :fail
python revision\train_backbone_v2.py --seed 1 --aug strong --out ckpt_rev\strong_s1 || goto :fail
python revision\train_backbone_v2.py --seed 2 --aug strong --out ckpt_rev\strong_s2 || goto :fail

REM --- augmentation ablation: third seed of the ORIGINAL light augmentation (seeds 0 and 1 already exist)
python revision\train_backbone_v2.py --seed 2 --aug light --epochs 30 --out ckpt_rev\light_s2 || goto :fail

REM --- R3 Method 2: sensitivity to lambda (confidence-loss weight); lambda=0.5 is strong_s0 above
python revision\train_backbone_v2.py --seed 0 --aug strong --lambda-conf 0.1 --out ckpt_rev\strong_s0_lam0.1 || goto :fail
python revision\train_backbone_v2.py --seed 0 --aug strong --lambda-conf 1.0 --out ckpt_rev\strong_s0_lam1.0 || goto :fail

REM --- fit the outcome-supervised gate on the VALIDATION split for every backbone
for %%B in (strong_s0 strong_s1 strong_s2 light_s2 strong_s0_lam0.1 strong_s0_lam1.0) do (
    python revision\fit_gate.py --backbone ckpt_rev\%%B\best.pt --out ckpt_rev\%%B\gate.joblib || goto :fail
)
REM original submitted backbones (light augmentation, seeds 0 and 1) -- copy into the same layout
if not exist ckpt_rev\light_s0 mkdir ckpt_rev\light_s0
if not exist ckpt_rev\light_s1 mkdir ckpt_rev\light_s1
copy /Y checkpoints_backbone\epoch_22.pt ckpt_rev\light_s0\best.pt
copy /Y gate.joblib ckpt_rev\light_s0\gate.joblib
REM !! check that epoch_25.pt is the best seed-1 checkpoint (paper: best epoch 25, val IoU 0.667)
copy /Y checkpoints_backbone_seed1\epoch_25.pt ckpt_rev\light_s1\best.pt
copy /Y gate_seed1.joblib ckpt_rev\light_s1\gate.joblib

echo PHASE 1 DONE
goto :eof
:fail
echo *** PHASE 1 FAILED at the last command above ***
exit /b 1
