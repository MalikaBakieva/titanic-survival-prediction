"""
Titanic Survival Prediction Web Application
Built with Flask + Logistic Regression (scikit-learn)
"""

import os
import sys
import json
import csv
import html
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Set UTF-8 stdout if possible on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

app = Flask(__name__)

# Security: Session secret key & Admin password
app.secret_key = os.environ.get('SECRET_KEY', 'titanic-malika-secure-2026-key')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'malika2026')

# ─── Model setup ────────────────────────────────────────────────────────────
MODEL = None
SCALER = None
FEATURE_NAMES = ["Pclass", "Sex", "Age", "FamilySize", "IsAlone", "Fare", "Embarked_Q", "Embarked_S"]

# Average fares per class (from the original Titanic dataset)
CLASS_FARES = {
    1: 84.15,   # 1-klass o'rtacha narxi
    2: 20.66,   # 2-klass o'rtacha narxi
    3: 13.68,   # 3-klass o'rtacha narxi
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "predictions.csv")
TITANIC_CSV = os.path.join(DATA_DIR, "titanic.csv")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


def train_model():
    """Train the logistic regression model using Titanic data."""
    global MODEL, SCALER

    df = None
    # 1. Try local data file first
    if os.path.exists(TITANIC_CSV):
        try:
            df = pd.read_csv(TITANIC_CSV)
        except Exception as e:
            print(f"[OGOHLANTIRISH] Mahalliy faylni o'qishda xatolik: {e}")

    # 2. Try online dataset if local not found
    if df is None:
        try:
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            df = pd.read_csv(url)
            # Cache locally for future offline use
            df.to_csv(TITANIC_CSV, index=False)
        except Exception as e:
            print(f"[XATOLIK] Internetdan yuklab bo'lmadi: {e}")
            return False

    # Data cleaning
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

    # Feature engineering
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # One-hot encode Embarked
    df['Embarked_Q'] = (df['Embarked'] == 'Q').astype(int)
    df['Embarked_S'] = (df['Embarked'] == 'S').astype(int)

    # Prepare features and target
    X = df[FEATURE_NAMES]
    y = df['Survived']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    SCALER = StandardScaler().fit(X_train)

    # Train model
    MODEL = LogisticRegression(max_iter=1000).fit(SCALER.transform(X_train), y_train)

    accuracy = MODEL.score(SCALER.transform(X_test), y_test)
    print(f"[TAYYOR] Model muvaffaqiyatli o'qitildi! Aniqlik (Accuracy): {accuracy * 100:.2f}%")

    return True


def predict_survival(pclass, sex, age, family_members, fare, embarked):
    """
    Predict survival probability using feature engineering.
    FamilySize = family_members + 1 (Oila hajmi = o'zi + oila a'zolari)
    IsAlone = 1 if family_size == 1 else 0 (Yolg'iz sayohat qilganligi)
    sex: 0=male, 1=female
    embarked: 'C', 'Q', or 'S'
    """
    if MODEL is None or SCALER is None:
        train_model()

    family_size = family_members + 1
    is_alone = 1 if family_size == 1 else 0
    embarked_q = 1 if embarked == 'Q' else 0
    embarked_s = 1 if embarked == 'S' else 0

    features_df = pd.DataFrame(
        [[pclass, sex, age, family_size, is_alone, fare, embarked_q, embarked_s]],
        columns=FEATURE_NAMES
    )
    scaled = SCALER.transform(features_df)
    probability = MODEL.predict_proba(scaled)[0][1]
    return round(probability * 100, 1)


def save_prediction(data):
    """Save prediction data to CSV file."""
    file_exists = os.path.exists(DATA_FILE)

    fieldnames = [
        'timestamp', 'name', 'sex', 'age', 'pclass', 'fare',
        'family_members', 'family_size', 'is_alone', 'embarked', 'survival_probability', 'ip_address'
    ]

    with open(DATA_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


def get_predictions_data():
    """Read all predictions from CSV."""
    if not os.path.exists(DATA_FILE):
        return []

    predictions = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append(row)

    return predictions


# ─── Public Routes ──────────────────────────────────────────────────────────
@app.route('/')
def index():
    """Main prediction page (Public)."""
    return render_template('index.html', class_fares=CLASS_FARES)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests (Public API)."""
    try:
        data = request.get_json()

        raw_name = data.get('name', '').strip()
        # XSS sanitization
        name = html.escape(raw_name[:50])

        sex = int(data['sex'])           # 0=male, 1=female
        age = float(data['age'])
        pclass = int(data['pclass'])
        fare = float(data['fare'])

        # Feature engineering: siblings + parents/children = family_members
        if 'family_members' in data:
            family_members = int(data['family_members'])
        else:
            family_members = int(data.get('sibsp', 0)) + int(data.get('parch', 0))

        family_size = family_members + 1
        is_alone = 1 if family_size == 1 else 0
        embarked = data.get('embarked', 'S')

        probability = predict_survival(pclass, sex, age, family_members, fare, embarked)

        # Anonymize IP address for user privacy (e.g. 192.168.*.*)
        raw_ip = request.remote_addr or '127.0.0.1'
        ip_parts = raw_ip.split('.')
        anonymized_ip = f"{ip_parts[0]}.{ip_parts[1]}.*.*" if len(ip_parts) == 4 else "anonim"

        # Save to dataset
        save_prediction({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'name': name if name else 'Anonim',
            'sex': 'Erkak' if sex == 0 else 'Ayol',
            'age': age,
            'pclass': pclass,
            'fare': fare,
            'family_members': family_members,
            'family_size': family_size,
            'is_alone': "Ha" if is_alone == 1 else "Yo'q",
            'embarked': embarked,
            'survival_probability': probability,
            'ip_address': anonymized_ip
        })

        return jsonify({
            'success': True,
            'probability': probability,
            'name': name if name else 'Sayohatchi'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ─── Secure Admin Routes ────────────────────────────────────────────────────
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    """Secure login for admin only."""
    error = None
    if request.method == 'POST':
        password = request.form.get('password', '')
        if password == ADMIN_PASSWORD:
            session['is_admin'] = True
            return redirect(url_for('stats'))
        else:
            error = "Noto'g'ri parol! Qaytadan urinib ko'ring."
    return render_template('admin_login.html', error=error)


@app.route('/admin-logout')
def admin_logout():
    """Logout admin session."""
    session.pop('is_admin', None)
    return redirect(url_for('index'))


@app.route('/stats')
def stats():
    """Show collected predictions data (Admin Only)."""
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))
    predictions = get_predictions_data()
    return render_template('stats.html', predictions=predictions)


@app.route('/api/stats')
def api_stats():
    """API endpoint for stats data (Admin Only)."""
    if not session.get('is_admin'):
        return jsonify({'error': 'Ruxsat berilmagan. Faqat admin uchun!'}), 403
    predictions = get_predictions_data()
    return jsonify({
        'total': len(predictions),
        'predictions': predictions
    })


@app.route('/download-dataset')
def download_dataset():
    """Download the collected predictions CSV file (Admin Only)."""
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))
    if os.path.exists(DATA_FILE):
        return send_file(DATA_FILE, as_attachment=True, download_name='titanic_user_predictions.csv')
    return "Dataset hali mavjud emas. Avval bashorat qilib ko'ring!", 404


# ─── Initialize ─────────────────────────────────────────────────────────────
print("Titanic Survival Predictor - Model yuklanmoqda...")
train_model()
print("Model muvaffaqiyatli ishga tushirildi!")


if __name__ == '__main__':
    # Local network or localhost
    app.run(debug=True, host='0.0.0.0', port=5000)
