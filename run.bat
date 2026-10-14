@echo off
chcp 65001 > nul
title Titanic Survival Predictor
echo ========================================================
echo        TITANIC BASHORAT LOYIHASI ISHGA TUSHMOQDA
echo ========================================================
echo.
echo Brauzerda ochiladigan manzil: http://127.0.0.1:5000
echo Serverni to'xtatish uchun: Ctrl + C tugmalarini bosing.
echo.

start "" "http://127.0.0.1:5000"
py app.py

pause
