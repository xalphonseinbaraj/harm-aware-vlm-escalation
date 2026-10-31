@echo off
REM ===================================================================================
REM PHASE 5 -- strengthening experiments (beyond the reviewers' requests)
REM   B: a medical VLM (Google MedGemma 4B) as a third escalation target, on the Kvasir test split
REM   A: cross-dataset generalisation on three external polyp datasets (other hospitals / devices)
REM Approx. 7 hours in total. Every run resumes if interrupted -- just run this file again.
REM
REM ONE-TIME PREPARATION
REM  1. Ollama must be running with E:\ollama_models  (Window A: set OLLAMA_MODELS=E:\ollama_models  then  ollama serve)
REM  2. ollama pull medgemma:4b                          (about 3.3 GB, saved to E:\ollama_models)
REM  3. Download PraNet's TestDataset (327 MB) in a browser:
REM       https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao/view?usp=sharing
REM     and unzip it so that these folders exist:
REM       E:\polyp_llm_seg\data\TestDataset\CVC-ClinicDB\images   and \masks
REM       E:\polyp_llm_seg\data\TestDataset\CVC-ColonDB\images    and \masks
REM       E:\polyp_llm_seg\data\TestDataset\ETIS-LaribPolypDB\images and \masks
REM ===================================================================================
cd /d E:\polyp_llm_seg
set TD=data\TestDataset
for %%D in (CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB) do (
    if not exist %TD%\%%D\images ( echo *** missing %TD%\%%D\images -- see step 3 above & goto :fail )
)

REM ---------- B: MedGemma on the Kvasir test split, 3 strong-augmentation seeds (joins the main Table 7 analysis)
for %%B in (strong_s0 strong_s1 strong_s2) do (
    python revision\eval_all.py --backbone ckpt_rev\%%B\best.pt --gate ckpt_rev\%%B\gate.joblib --vlm medgemma4b_p1 --out results_rev\%%B__medgemma4b_p1.jsonl --save-masks masks_rev\%%B__medgemma4b_p1 || goto :fail
)

REM ---------- A1: matched baselines on the external sets (backbone only, fast)
for %%D in (CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB) do (
    for %%M in (unet_r34 unetpp_r34 deeplabv3p_r50 segformer_mitb2 fpn_pvtv2b2) do (
        python revision\eval_all.py --backbone ckpt_rev\base_%%M\best.pt --data-root %TD%\%%D --out results_ext\base_%%D_%%M.jsonl || goto :fail
    )
)

REM ---------- A2: our backbones + gate + Qwen2.5-VL-7B on the external sets (trained on Kvasir only; no re-training, no re-fitting)
for %%D in (CVC-ColonDB ETIS-LaribPolypDB CVC-ClinicDB) do (
    for %%B in (strong_s0 strong_s1 strong_s2) do (
        python revision\eval_all.py --backbone ckpt_rev\%%B\best.pt --gate ckpt_rev\%%B\gate.joblib --vlm qwen7b_p3 --data-root %TD%\%%D --out results_ext\%%D_%%B__qwen7b_p3.jsonl || goto :fail
    )
)

REM ---------- A3: MedGemma on the largest external set, seed 0
python revision\eval_all.py --backbone ckpt_rev\strong_s0\best.pt --gate ckpt_rev\strong_s0\gate.joblib --vlm medgemma4b_p1 --data-root %TD%\CVC-ColonDB --out results_ext\CVC-ColonDB_strong_s0__medgemma4b_p1.jsonl || goto :fail

REM ---------- analysis
python revision\analyze.py --dir results_rev || goto :fail
python revision\analyze.py --dir results_ext || goto :fail
echo.
echo PHASE 5 DONE
goto :eof
:fail
echo *** PHASE 5 STOPPED -- fix the message above and run this file again (finished images are kept) ***
exit /b 1
