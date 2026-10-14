# 🚢 Titanic ML Web Loyihasi: Noldan Jonli Havolagacha Mukammal O'quv Qo'llanma

Ushbu qo'llanma **Machine Learning (Sun'iy intellekt)** modelini Jupyter Notebook'dagi oddiy tahlildan boshlab, real foydalanuvchilar kirib ishlatishi mumkin bo'lgan **zamonaviy, ikki tilli, xavfsiz va 24/7 internetda ishlaydigan veb-ilova** holatiga keltirishning to'liq yo'lini qadamma-qadam tushuntirib beradi.

Qo'llanma har qanday yangi o'rganuvchi mustaqil ravishda qayta qura oladigan tarzda, har bir kod va har bir buyruqning batafsil izohi bilan yozilgan.

---

## 📑 Mundarija
1. [Loyiha Maqsadi va Dastlabki Holat](#1-loyiha-maqsadi-va-dastlabki-holat)
2. [Loyiha Arxitekturasi va Papkalar Strukturasi](#2-loyiha-arxitekturasi-va-papkalar-strukturasi)
3. [Kerakli Kutubxonalarni O'rnatish](#3-kerakli-kutubxonalarni-ornatish)
4. [Backend Dasturlash (app.py) — Har bir kod tahlili](#4-backend-dasturlash-apppy--har-bir-kod-tahlili)
5. [Frontend (Interfeys) Yaratish — HTML, CSS va JS](#5-frontend-interfeys-yaratish--html-css-va-js)
6. [Xavfsizlik va Admin Panel](#6-xavfsizlik-va-admin-panel)
7. [Telegram Bot Integratsiyasi (Ma'lumotlar yo'qolmasligi uchun)](#7-telegram-bot-integratsiyasi-malumotlar-yoqolmasligi-uchun)
8. [GitHub va Render.com orqali 24/7 Jonli Havola Olish](#8-github-va-rendercom-orqali-247-jonli-havola-olish)
9. [Xulosa va Olingan Ko'nikmalar](#9-xulosa-va-olingan-konikmalar)

---

## 1. Loyiha Maqsadi va Dastlabki Holat

### Dastlab bizda nima bor edi?
Bizda Google Colab yoki Jupyter Notebook'da yaratilgan `lessoon33.ipynb` fayli bor edi. Unda:
* 1912-yilgi Titanik yo'lovchilari to'g'risidagi tarixiy ma'lumotlar to'plami (`titanic.csv`);
* Logistik regressiya (Logistic Regression) modeli;
* Sigmoid matematik funksiyasi ($p = \frac{1}{1 + e^{-z}}$);
* Modelning aniqlik darajasi (Accuracy ~79%) bor edi.

### Asosiy Muammo:
Notebook'dagi kod faqat dasturchining o'z kompyuterida ishlaydi. Oddiy odamlar yoki rekruterlar undan foydalana olmaydi. Bizga:
1. Chiroyli, estetik, ikki tilli (O'zbekcha / Inglizcha) dizayn;
2. Istalgan odam yoshi, jinsi, chipta klassini kiritganda darhol omon qolish ehtimolligini (%) hisoblab beruvchi tizim;
3. Kiritilgan ma'lumotlarni bazaga yig'ib boruvchi mexanizm;
4. Natijalar o'chib ketmasligi uchun Telegramga avtomatik yetkazuvchi bot;
5. Internetda hamma uchun ochiq 24/7 havola (link) kerak edi.

---

## 2. Loyiha Arxitekturasi va Papkalar Strukturasi

Avvalo loyihamiz tartibli bo'lishi uchun quyidagi papkalar iyerarxiyasini quramiz:

```text
titanic_app/
│
├── app.py                  # Asosiy Python server (Flask backend + ML model)
├── requirements.txt        # Serverga kerakli Python kutubxonalari ro'yxati
├── Procfile                # Render/Heroku serveri ilovani qanday yurgizish buyrug'i
├── .env                    # Maxfiy kalitlar (Token, parol — gitga yuklanmaydi)
├── .gitignore              # Gitga yuklanmasligi kerak bo'lgan fayllar ro'yxati
│
├── data/
│   ├── titanic.csv         # Model o'rganishi uchun tarixiy dataset
│   └── predictions.csv     # Sayt foydalanuvchilari kiritgan ma'lumotlar bazasi
│
├── static/
│   └── css/
│       └── style.css       # Premium Dark Ocean dizayni va animatsiyalar
│
└── templates/
    ├── index.html          # Asosiy bashorat sahifasi (Ikki tilli UI)
    ├── admin_login.html    # Parol bilan himoyalangan admin kirish sahifasi
    └── stats.html          # Yig'ilgan natijalar statistikasi va CSV yuklab olish
```

---

## 3. Kerakli Kutubxonalarni O'rnatish

Loyihani yurgizish uchun kompyuterga minimal va eng kerakli kutubxonalarni o'rnatamiz:

Terminalda (PowerShell yoki CMD) quyidagi buyruq bajariladi:
```bash
pip install flask scikit-learn pandas numpy gunicorn
```

### Nega aynan shu kutubxonalar?
* **`flask`** — Yengil, tez va qulay Python veb-freymvorki. Veb-sayt sahifalarini ochish va API so'rovlarni boshqarish uchun xizmat qiladi.
* **`scikit-learn`** — Machine Learning sohasining eng asosiy kutubxonasi. Logistik regressiya modeli (`LogisticRegression`) va ma'lumotlarni normallashtirish (`StandardScaler`) aynan shundan olinadi.
* **`pandas`** va **`numpy`** — Jadval ma'lumotlarini (CSV) o'qish, tozalash va massivlar bilan hisob-kitob qilish uchun.
* **`gunicorn`** — Python ilovasini internet serverida (masalan, Render.com da) professional darajada yurgizib beruvchi WSGI HTTP server.

Server uchun `requirements.txt` faylini yaratamiz:
```text
flask>=3.0.0
scikit-learn>=1.4.0
pandas>=2.1.0
numpy>=1.26.0
gunicorn>=21.0.0
```

---

## 4. Backend Dasturlash (`app.py`) — Har bir kod tahlili

`app.py` — butun loyihaning yuragi. U o'zida modelni o'qitish, veb-so'rovlarni qabul qilish, hisoblash va natijani chiqarishni birlashtiradi.

Keling, uni bosqichma-bosqich ko'rib chiqamiz:

### 1-qadam: Kutubxonalarni import qilish va sozlamalar
```python
import os
import json
import csv
import html
import urllib.request
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Maxfiy kalitlar (Parol va Telegram sozlamalari)
app.secret_key = os.environ.get('SECRET_KEY', 'titanic-malika-secure-2026-key')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'malika2026')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '...')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '...')
```
* **Izoh:** `html` moduli kiberxavfsizlik (XSS hujumlari)ning oldini oladi, `urllib.request` esa ortiqcha kutubxonasiz Telegram API ga tezkor xabar yuboradi.

---

### 2-qadam: Machine Learning Modelini o'qitish funksiyasi (`train_model`)
Model har safar server yonganda avtomatik ravishda Titanik datasetini o'qib, xotirada o'qitiladi:

```python
MODEL = None
SCALER = None
FEATURE_NAMES = ["Pclass", "Sex", "Age", "FamilySize", "IsAlone", "Fare", "Embarked_Q", "Embarked_S"]

def train_model():
    global MODEL, SCALER
    
    # 1. Datasetni yuklash (agar kompyuterda bo'lsa mahalliy, bo'lmasa internetdan)
    if os.path.exists(TITANIC_CSV):
        df = pd.read_csv(TITANIC_CSV)
    else:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
        df.to_csv(TITANIC_CSV, index=False)

    # 2. Ma'lumotlarni tozalash (Data Cleaning)
    df['Age'] = df['Age'].fillna(df['Age'].median())      # Bo'sh yoshlarni o'rtacha qiymat bilan to'ldirish
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())    # Bo'sh narxlarni to'ldirish
    df['Embarked'] = df['Embarked'].fillna('S')           # Portni 'S' bilan to'ldirish

    # 3. Feature Engineering (Yangi belgilar yaratish)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})    # Jinsni sonlarga o'tkazish
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1      # Oila a'zolari = Aka-uka + Ota-ona + O'zi
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)   # Yolg'izmi? (1 = Ha, 0 = Yo'q)

    # One-Hot Encoding (Portlarni ikkilik sonlarga ajratish)
    df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
    df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)

    X = df[FEATURE_NAMES]
    y = df['Survived']

    # 4. Train / Test ga ajratish (80% o'rganish, 20% sinov)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Masshtablash (Scaling)
    SCALER = StandardScaler().fit(X_train)

    # 6. Logistik Regressiyani o'qitish
    MODEL = LogisticRegression(max_iter=1000).fit(SCALER.transform(X_train), y_train)
```
* **Nega Feature Engineering qildik?** Titanikda yolg'iz sayohat qilganlar va katta oilalilarning omon qolish ehtimoli keskin farq qilgan. `SibSp` (aka-uka/er-xotin) va `Parch` (ota-ona/farzand) larni birlashtirib, `FamilySize` (oila hajmi) ni hosil qilish model aniqligini sezilarli oshiradi.

---

### 3-qadam: Bashorat qilish funksiyasi (`predict_survival`)
Foydalanuvchi ma'lumot kiritganda, uning omon qolish ehtimolligi (%) qanday hisoblanadi?

```python
def predict_survival(pclass, sex, age, family_members, fare, embarked):
    family_size = family_members + 1
    is_alone = 1 if family_size == 1 else 0
    embarked_q = 1 if embarked == 'Q' else 0
    embarked_s = 1 if embarked == 'S' else 0

    features_df = pd.DataFrame(
        [[pclass, sex, age, family_size, is_alone, fare, embarked_q, embarked_s]],
        columns=FEATURE_NAMES
    )
    scaled = SCALER.transform(features_df)
    
    # Model 0 va 1 oraliqdagi ehtimollikni beradi, biz uni foizga (%) aylantiramiz
    probability = MODEL.predict_proba(scaled)[0][1]
    return round(probability * 100, 1)
```

---

### 4-qadam: API marshruti (`/predict`) va Xavfsizlik
Foydalanuvchi "Bashorat Qilish" tugmasini bosganda Frontend JavaScript shu manzilga JSON ma'lumot jo'natadi:

```python
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # XSS hujumiga qarshi sanitizatsiya (Zararli scriptlarni zararsizlantirish)
    raw_name = data.get('name', '').strip()
    name = html.escape(raw_name[:50])

    sex = int(data['sex'])
    age = float(data['age'])
    pclass = int(data['pclass'])
    fare = float(data['fare'])
    family_members = int(data.get('family_members', 0))
    embarked = data.get('embarked', 'S')

    # Ehtimollikni hisoblaymiz
    probability = predict_survival(pclass, sex, age, family_members, fare, embarked)

    # Foydalanuvchi shaxsiy daxlsizligi uchun IP manzilni yashirish (192.168.*.*)
    raw_ip = request.remote_addr or '127.0.0.1'
    ip_parts = raw_ip.split('.')
    anonymized_ip = f"{ip_parts[0]}.{ip_parts[1]}.*.*" if len(ip_parts) == 4 else "anonim"

    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'name': name if name else 'Anonim',
        'sex': 'Erkak' if sex == 0 else 'Ayol',
        'age': age,
        'pclass': pclass,
        'fare': fare,
        'family_members': family_members,
        'embarked': embarked,
        'survival_probability': probability,
        'ip_address': anonymized_ip
    }

    # 1. Datasetga saqlash
    save_prediction(record)

    # 2. Telegramga tezkor bildirishnoma yuborish
    send_telegram_notification(record)

    return jsonify({'success': True, 'probability': probability, 'name': name})
```

---

## 5. Frontend (Interfeys) Yaratish — HTML, CSS va JS

Interfeys foydalanuvchida birinchi ko'rishdayoq katta taassurot qoldirishi kerak. Shuning uchun:

1. **Dark Ocean Dizayni:** Chuqur okean fonida suzuvchi to'lqinlar va yulduzlar animatsiyasi.
2. **Glassmorphism (Shisha effekti):** Qorong'i, xiralashtirilgan (`backdrop-filter: blur(16px)`) kartalar.
3. **Avtomatik narx belgilash:** 
   1-klass tanlanganda avtomatik ravishda **$84.15**, 2-klassda **$20.66**, 3-klassda **$13.68** belgilanadi.
4. **SVG Circular Gauge:** Omon qolish ehtimoli oddiy son emas, animatsiyali doira shaklida aylanib chiqadi:
   * **Yashil** (≥ 70%): Omon qolish imkoniyati yuqori;
   * **Sariq** (40% - 69%): 50/50 o'rtacha holat;
   * **Qizil** (< 40%): Xavf yuqori.
5. **Ikki tilli xalqaro interfeys:** Har bir maydon o'zbekcha va inglizcha yozuvlar bilan ta'minlangan (`Ism / Full Name`, `Jins / Gender`, `Yosh / Age`).

---

## 6. Xavfsizlik va Admin Panel

Foydalanuvchilar o'z ma'lumotlarini kiritganda to'plangan dataset begona odamlar qo'liga tushmasligi kerak. Buning uchun tizimga maxsus himoya o'rnatildi:

* `/stats` va `/download-dataset` yo'llariga begona odam kirsa, tizim uni darhol `/admin-login` ga qaytaradi.
* Faqatgina admin paroli (`malika2026`) kiritilgandagina sessiya ochiladi.
* Admin panelda:
  * Jami test qilganlar soni;
  * Omon qolish ehtimoli 50% dan yuqori bo'lganlar statistikasi;
  * Barcha natijalar jadvali;
  * To'plangan ma'lumotlarni **`.csv` fayl qilib yuklab olish (Download)** tugmasi mavjud.

---

## 7. Telegram Bot Integratsiyasi (Ma'lumotlar yo'qolmasligi uchun)

### Muammo nimada edi?
Render.com kabi bepul bulutli serverlar **"Ephemeral Disk"** (vaqtinchalik virtual disk) da ishlaydi. Ya'ni saytga 15 daqiqa hech kim kirmasa, server o'chib "uyquga ketadi". Yangi odam kirib uni uyg'otganida esa, fayllar tozalanib, GitHub'dagi dastlabki holatiga qaytib qoladi. Natijada CSV faylga yozilgan ma'lumotlar o'chib ketishi mumkin.

### Mukammal Yechim:
Har safar kimdir saytda bashorat qilishi bilan ma'lumot soniyasidayoq **Telegram Botingizga** kelib tushadi!

```python
def send_telegram_notification(data):
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        return False

    status_icon = "🟢" if data['survival_probability'] >= 50 else "🔴"

    msg = (
        f"🚢 <b>Yangi Titanic Bashorati!</b>\n\n"
        f"👤 <b>Ism:</b> {data['name']}\n"
        f"⚧ <b>Jins:</b> {data['sex']}\n"
        f"🎂 <b>Yosh:</b> {data['age']}\n"
        f"🎫 <b>Klass:</b> {data['pclass']}-klass (${data['fare']})\n"
        f"👨‍👩‍👧‍👦 <b>Oila a'zolari:</b> {data['family_members']} ta\n"
        f"🏖 <b>Port:</b> {data['embarked']}\n"
        f"🔮 <b>Ehtimollik:</b> <b>{data['survival_probability']}%</b> {status_icon}\n"
        f"⏱ <b>Vaqt:</b> {data['timestamp']}"
    )

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}).encode('utf-8')
    req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status == 200
```

### Buni qanday ulaymiz?
1. Telegramda **`@BotFather`** ga kirib `/newbot` buyrug'i bilan bot ochiladi va `Token` olinadi.
2. **`@userinfobot`** ga kirib shaxsiy `Chat ID` raqami olinadi.
3. Botga kirib bir marta `/start` bosiladi.
4. Token va Chat ID `app.py` ga joylanadi. Endi server o'chsa ham, barcha ma'lumotlar telefoningizda saqlanadi!

---

## 8. GitHub va Render.com orqali 24/7 Jonli Havola Olish

Loyihani internetga chiqarish eng hayajonli qadamdir.

### 1-qadam: `Procfile` yaratish
Render serveriga ilovani qanday yurgizishni aytish uchun loyiha boshiga `Procfile` nomli fayl joylanadi:
```text
web: gunicorn app:app
```

### 2-qadam: GitHub'ga yuklash
Terminalda quyidagi ketma-ketlik bajariladi:
```bash
git init
git add .
git commit -m "Titanic ML web app with Telegram bot and bilingual UI"
git branch -M main
git remote add origin https://github.com/USERNAME/REPO_NAME.git
git push -u origin main
```

### 3-qadam: Render.com da bepul server ochish
1. [dashboard.render.com](https://dashboard.render.com) saytiga kirib, **"New +" -> "Web Service"** tanlanadi.
2. GitHub repozitoriyangiz tanlanadi.
3. Sozlamalar:
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `gunicorn app:app`
   * **Plan:** `Free` (Bepul)
4. **"Deploy Web Service"** tugmasi bosiladi.

2-3 daqiqadan so'ng Render sizga quyidagicha doimiy xalqaro HTTPS havola beradi:
👉 `https://titanic-survival-prediction-xxxx.onrender.com`

Ushbu havolani LinkedIn, Telegram yoki rezyumengizga qo'yishingiz mumkin — u kompyuteringiz o'chiq bo'lsa ham 24/7 ishlab turadi!

---

## 9. Xulosa va Olingan Ko'nikmalar

Ushbu loyihani bajarish orqali siz:
1. **Data Science & ML:** Ma'lumotlarni tozalash, yangi belgilar yasash (Feature Engineering) va Logistik Regressiyani to'g'ri o'qitishni;
2. **Web Development:** Flask bilan to'liq ishlaydigan RESTful veb-ilova yaratishni;
3. **Frontend UI/UX:** Zamonaviy Dark Ocean dizayn, animatsiyalar va ikki tilli (O'zbekcha / Inglizcha) moslashuvchan interfeys qurishni;
4. **DevOps & Cloud:** Git, GitHub va Render.com orqali ilovalarni internetga chiqarishni (Deployment);
5. **Real-time Alerting:** Telegram Bot API orqali hodisalarni avtomatik monitoring qilishni o'rgandingiz.

Bu loyiha har qanday Data Science yoki Dasturlash portfoliosi uchun faxr bilan ko'rsatsa arziydigan, to'liq yakunlangan professional ish hisoblanadi! 🚀
