from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
import pickle
import numpy as np
import pandas as pd
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "soil_health_secret_key"

# Database Configuration (WAMP Defaults)
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'soil_health_db'
}

def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Load ML Models
MODEL_PATH = "models"
try:
    with open(os.path.join(MODEL_PATH, 'model_soil.pkl'), 'rb') as f:
        model_soil = pickle.load(f)
    with open(os.path.join(MODEL_PATH, 'model_fertilizer.pkl'), 'rb') as f:
        model_fert = pickle.load(f)
    with open(os.path.join(MODEL_PATH, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_PATH, 'encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")

# --- Helper Functions ---

def calculate_score(n, p, k):
    # Normalized score logic (simplified)
    # Average of N,P,K relative to a healthy baseline (e.g. 50, 40, 40)
    score = (min(n, 100) + min(p, 100) + min(k, 100)) / 3
    return round(score, 1)

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        # --- Static Admin Logic ---
        if email == 'admin' and password == 'admin':
            session['user_id'] = 0  # Special ID for static admin
            session['user_name'] = "System Administrator"
            session['role'] = 'admin'
            flash("Logged in as Administrator", "success")
            return redirect(url_for('admin_dashboard'))

        # --- Farmer/User DB Logic ---
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            user = cursor.fetchone()
            conn.close()
            
            if user:
                if user['password'] == password or check_password_hash(user['password'], password):
                    session['user_id'] = user['id']
                    session['user_name'] = user['name']
                    session['role'] = user['role']
                    flash("Login successful!", "success")
                    return redirect(url_for('dashboard'))
            
            flash("Invalid email or password", "danger")
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", 
                               (name, email, password))
                conn.commit()
                flash("Registration successful! Please login.", "success")
                return redirect(url_for('login'))
            except mysql.connector.Error as err:
                flash(f"Error: {err}", "danger")
            finally:
                conn.close()
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Simple stats for dashboard
    conn = get_db_connection()
    stats = {'total_tests': 0}
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM soil_tests WHERE user_id = %s", (session['user_id'],))
        stats['total_tests'] = cursor.fetchone()[0]
        conn.close()
        
    return render_template('dashboard.html', stats=stats)

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        # Get data from form
        n = float(request.form['nitrogen'])
        p = float(request.form['phosphorous'])
        k = float(request.form['potassium'])
        temp = float(request.form['temperature'])
        hum = float(request.form['humidity'])
        moist = float(request.form['moisture'])
        soil_type = request.form['soil_type']
        crop_type = request.form['crop_type']
        
        # Prepare input for ML
        # Logic matches main.py: [temp, hum, moist, soil_enc, crop_enc, n, k, p]
        input_data = pd.DataFrame([[temp, hum, moist, 0, 0, n, k, p]], 
                                columns=['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'])
        
        input_data['Soil Type'] = encoders['Soil Type'].transform([soil_type])[0]
        input_data['Crop Type'] = encoders['Crop Type'].transform([crop_type])[0]
        
        # Scale
        numeric_cols = ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
        input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
        
        # Features list used during training
        features_h = ['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']
        
        # 1. Predict Health
        health_enc = model_soil.predict(input_data[features_h])[0]
        health_label = encoders['Soil_Health'].inverse_transform([health_enc])[0]
        
        # 2. Predict Fertilizer
        input_data['Soil_Health_Encoded'] = health_enc
        features_f = features_h + ['Soil_Health_Encoded']
        fert_enc = model_fert.predict(input_data[features_f])[0]
        fert_label = encoders['Fertilizer'].inverse_transform([fert_enc])[0]
        
        score = calculate_score(n, p, k)
        
        # Save to DB
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO soil_tests 
                (user_id, nitrogen, phosphorous, potassium, temperature, humidity, moisture, soil_type, crop_type, soil_health, fertilizer, score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (session['user_id'], n, p, k, temp, hum, moist, soil_type, crop_type, health_label, fert_label, score))
            conn.commit()
            conn.close()
            
        return render_template('result.html', result={
            'health': health_label,
            'fertilizer': fert_label,
            'score': score,
            'data': request.form
        })
        
    except Exception as e:
        flash(f"Error in prediction: {e}", "danger")
        return redirect(url_for('dashboard'))

@app.route('/history')
def history():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    tests = []
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM soil_tests WHERE user_id = %s ORDER BY created_at DESC", (session['user_id'],))
        tests = cursor.fetchall()
        conn.close()
    return render_template('history.html', tests=tests)

@app.route('/admin')
def admin_dashboard():
    if session.get('role') != 'admin':
        flash("Access Denied", "danger")
        return redirect(url_for('dashboard'))
    
    conn = get_db_connection()
    stats = {}
    recent_activity = []
    if conn:
        cursor = conn.cursor(dictionary=True)
        # KPI 1: Total Users
        cursor.execute("SELECT COUNT(*) as count FROM users WHERE role = 'farmer'")
        stats['total_users'] = cursor.fetchone()['count']
        
        # KPI 2: Total Tests
        cursor.execute("SELECT COUNT(*) as count FROM soil_tests")
        stats['total_tests'] = cursor.fetchone()['count']
        
        # KPI 3: Avg Score
        cursor.execute("SELECT AVG(score) as avg FROM soil_tests")
        res_avg = cursor.fetchone()['avg']
        stats['avg_score'] = round(res_avg if res_avg else 0, 1)
        
        # KPI 4: Health Index (% of Good Soil)
        cursor.execute("SELECT COUNT(*) as count FROM soil_tests WHERE soil_health = 'Good'")
        good_count = cursor.fetchone()['count']
        stats['health_index'] = round((good_count / stats['total_tests'] * 100), 1) if stats['total_tests'] > 0 else 0
        
        # Recent Activity
        cursor.execute("""
            SELECT u.name, t.soil_health, t.fertilizer, t.created_at, t.crop_type 
            FROM users u JOIN soil_tests t ON u.id = t.user_id 
            ORDER BY t.created_at DESC LIMIT 6
        """)
        recent_activity = cursor.fetchall()
        conn.close()
        
    return render_template('admin.html', view='home', stats=stats, recent=recent_activity)

@app.route('/admin/users')
def admin_users():
    if session.get('role') != 'admin':
        return redirect(url_for('dashboard'))
    
    conn = get_db_connection()
    users = []
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()
        conn.close()
    return render_template('admin.html', view='users', users=users)

@app.route('/admin/user/add', methods=['POST'])
def add_user():
    if session.get('role') != 'admin':
        return redirect(url_for('dashboard'))
    
    name = request.form['name']
    email = request.form['email']
    password = generate_password_hash('password123') # Default password
    role = request.form['role']
    
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (name, email, password, role) VALUES (%s, %s, %s, %s)", 
                           (name, email, password, role))
            conn.commit()
            flash("User added successfully!", "success")
        except Exception as e:
            flash(f"Error: {e}", "danger")
        finally:
            conn.close()
    return redirect(url_for('admin_users'))

@app.route('/admin/user/delete/<int:id>')
def delete_user(id):
    if session.get('role') != 'admin':
        return redirect(url_for('dashboard'))
    
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = %s", (id,))
        conn.commit()
        conn.close()
        flash("User deleted successfully!", "success")
    return redirect(url_for('admin_users'))

@app.route('/admin/train')
def admin_train():
    if session.get('role') != 'admin':
        return redirect(url_for('dashboard'))
    return render_template('admin.html', view='train')

if __name__ == '__main__':
    app.run(debug=True)
