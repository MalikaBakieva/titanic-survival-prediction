@echo off
chcp 65001 > nul
title Titanic Public Link Generator
echo ========================================================
echo   TITANIC BASHORAT — OMMANI INTERNET HAVOLASI
echo ========================================================
echo.
echo 1. Flask server tekshirilmoqda va ishga tushirilmoqda...
start /b py app.py
timeout /t 3 /nobreak > nul
echo.
echo 2. Dunyoga ochiq bepul HTTPS havola ulanmoqda...
echo.
echo [DIQQAT] Quyida "https://....lhr.life" ko'rinishidagi havola paydo bo'ladi.
echo O'sha havolani ijtimoiy tarmoqlarga yoki do'stlaringizga yuborishingiz mumkin!
echo.
ssh -o StrictHostKeyChecking=no -R 80:127.0.0.1:5000 nokey@localhost.run
pause
