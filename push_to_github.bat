@echo off
REM Uploads this folder to https://github.com/xalphonseinbaraj/harm-aware-vlm-escalation
cd /d E:\polyp_llm_seg\github_release
git --version || (echo *** Git is not installed: install it from https://git-scm.com/download/win & pause & exit /b 1)
git config --global user.name >NUL 2>&1 || git config --global user.name "Alphonse Inbaraj Xavier"
git config --global user.email >NUL 2>&1 || git config --global user.email "xalphonse@isu.edu.tw"
git config --global --add safe.directory E:/polyp_llm_seg/github_release
git init
git add .
git commit -m "Code, gates and result logs for the BSPC revision"
git branch -M main
git remote remove origin >NUL 2>&1
git remote add origin https://github.com/xalphonseinbaraj/harm-aware-vlm-escalation.git
git push -u origin main
echo.
echo === FINISHED: check the messages above ===
pause
