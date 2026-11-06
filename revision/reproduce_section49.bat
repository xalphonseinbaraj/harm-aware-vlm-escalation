@echo off
REM ===================================================================================
REM Reproduce the reported numbers of Section 4.9 from the saved per-image logs (no GPU, no MedSAM, no internet).
REM Each step prints the values reported in the manuscript; compare them with REPRODUCE_EXPECTED.txt.
REM ===================================================================================
cd /d "%~dp0.."
echo ===== 1. Implementation checks of Algorithm 2 =====
python revision\test_algorithm2.py || goto :fail
echo ===== 2. PolypGen primary analysis, per-centre results, Holm-adjusted p =====
python revision\polypgen_confirm.py || goto :fail
python -c "import json;r=json.load(open('results_ext/polypgen_confirm.json'));ps=sorted((v['p'],k) for k,v in r.items() if k.startswith('strong|C'));m=len(ps);run=0;[print(k,'p=%%.3g'%%p,'Holm=%%.3g'%%(run:=max(run,min(1,(m-i)*p)))) for i,(p,k) in enumerate(ps)]" || goto :fail
echo ===== 3. Learned acceptance on every dataset =====
python revision\learned_acceptance.py || goto :fail
echo ===== 4. Gated-only fitting =====
python revision\acceptance_population.py || goto :fail
echo ===== 5. Simpler rules =====
python revision\acceptance_baselines.py || goto :fail
echo ===== 6. Safety and cluster-aware intervals =====
python revision\safety_and_clustering.py || goto :fail
echo ===== 7. Image-grouped cross-validation =====
python revision\acceptance_grouped_cv.py || goto :fail
echo ===== 8. Agreement and calibration of the two fitting populations =====
python revision\acceptance_agreement.py || goto :fail
echo.
echo REPRODUCTION DONE
goto :eof
:fail
echo *** STOPPED -- see the message above ***
exit /b 1
