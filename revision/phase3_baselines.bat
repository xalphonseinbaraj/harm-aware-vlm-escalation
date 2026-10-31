@echo off
REM ===================================================================================
REM PHASE 3 -- matched baselines (R1 c6) + qualitative masks (R1 c7) + efficiency (R3 Results 3)
REM   pip install segmentation-models-pytorch timm        (once)
REM Approx. 40-90 min per baseline on RTX 4050. If a model runs out of GPU memory, lower --batch to 2.
REM ===================================================================================
cd /d E:\polyp_llm_seg

python revision\train_baselines.py --model unet_r34        --out ckpt_rev\base_unet_r34        || goto :fail
python revision\train_baselines.py --model unetpp_r34      --out ckpt_rev\base_unetpp_r34      || goto :fail
python revision\train_baselines.py --model deeplabv3p_r50  --out ckpt_rev\base_deeplabv3p_r50  || goto :fail
python revision\train_baselines.py --model segformer_mitb2 --out ckpt_rev\base_segformer_mitb2 --batch 4 || goto :fail
python revision\train_baselines.py --model fpn_pvtv2b2     --out ckpt_rev\base_fpn_pvtv2b2     --batch 4 || echo (PVTv2 baseline optional -- skipped)

for %%M in (unet_r34 unetpp_r34 deeplabv3p_r50 segformer_mitb2 fpn_pvtv2b2) do (
    if exist ckpt_rev\base_%%M\best.pt python revision\eval_all.py --backbone ckpt_rev\base_%%M\best.pt --out results_rev\base_%%M.jsonl --save-masks masks_rev\base_%%M
)

REM backbone-agnostic check (R1 c2/c3): the SAME gate + escalation on top of the strongest baseline
python revision\fit_gate.py --backbone ckpt_rev\base_segformer_mitb2\best.pt --out ckpt_rev\base_segformer_mitb2\gate.joblib || goto :fail
python revision\eval_all.py --backbone ckpt_rev\base_segformer_mitb2\best.pt --gate ckpt_rev\base_segformer_mitb2\gate.joblib --vlm qwen7b_p3 --out results_rev\segformer_s0__qwen7b_p3.jsonl || goto :fail
python revision\eval_all.py --backbone ckpt_rev\base_segformer_mitb2\best.pt --gate ckpt_rev\base_segformer_mitb2\gate.joblib --vlm llama11b_p1_nim --workers 6 --out results_rev\segformer_s0__llama11b_p1_nim.jsonl || goto :fail

set CK=ckpt_rev\strong_s0\best.pt ckpt_rev\light_s0\best.pt
for %%M in (unet_r34 unetpp_r34 deeplabv3p_r50 segformer_mitb2 fpn_pvtv2b2) do if exist ckpt_rev\base_%%M\best.pt call set CK=%%CK%% ckpt_rev\base_%%M\best.pt
python revision\efficiency.py --ckpts %CK% --out results_rev\efficiency.json || goto :fail
echo PHASE 3 DONE
goto :eof
:fail
echo *** PHASE 3 FAILED ***
exit /b 1
