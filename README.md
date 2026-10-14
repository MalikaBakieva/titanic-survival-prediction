# 🚢 Titanic Survival Prediction Web Application

Titanic halokatida yo'lovchilarning tirik qolish ehtimolligini hisoblab beruvchi interaktiv sun'iy intellekt (Machine Learning) veb-loyihasi.

## 📌 Asosiy xususiyatlar
- **ML Modeli:** Logistik Regressiya (`scikit-learn`) — aniqlik darajasi ~80%.
- **Feature Engineering:** `SibSp` va `Parch` parametrlari birlashtirilib, `FamilySize` (oila hajmi) va `IsAlone` (yolg'iz sayohatchi) ko'rsatkichlari hosil qilingan.
- **Dinamik narx:** Tanlangan chipta klassi (1, 2, 3-klass) ga qarab o'rtacha chipta narxi avtomatik belgilanadi.
- **Dataset yig'ish:** Foydalanuvchilarning barcha bashoratlari doimiy ravishda datasetga saqlab boriladi.
- **Admin Xavfsizlik:** Yig'ilgan statistika va CSV faylni yuklab olish faqat loyiha egasi (Admin) uchun parol orqali himoyalangan.
- **Telegram Share:** Natijani ijtimoiy tarmoqlarda do'stlar bilan ulashish imkoniyati.

## 🛠 Texnologiyalar
- **Backend:** Python, Flask, Gunicorn
- **Machine Learning:** Scikit-Learn, Pandas, NumPy
- **Frontend:** Vanilla HTML5, CSS3 (Dark Ocean Premium dizayn), JavaScript

## 🚀 Mahalliy ishga tushirish
1. Kutubxonalarni o'rnatish:
```bash
pip install -r requirements.txt
```
2. Loyihani ishga tushirish:
```bash
python app.py
```
Brauzerda `http://127.0.0.1:5000` manzilini oching.
