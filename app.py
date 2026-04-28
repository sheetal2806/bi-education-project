from flask import Flask, render_template, request, send_file
import pickle
import pandas as pd
import numpy as np
import sqlite3
import io
from database_setup import create_tables

app = Flask(__name__)

# Load model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

create_tables()

def classify_risk(score):
    if score < 50:
        return "High"
    elif score < 70:
        return "Medium"
    else:
        return "Low"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_single', methods=['POST'])
def predict_single():
    name = request.form['student_name']
    study_hrs = float(request.form['study_hours'])
    past_marks = float(request.form['past_marks'])
    attendance = float(request.form['attendance'])

    features = np.array([[study_hrs, past_marks, attendance]])
    prediction = model.predict(features)[0]
    prediction = max(0, min(100, prediction))
    risk = classify_risk(prediction)

    conn = sqlite3.connect('education_system.db')
    cur = conn.cursor()
    cur.execute("INSERT INTO students (student_name, daily_study_hours, previous_marks, attendance_rate) VALUES (?,?,?,?)",
                (name, study_hrs, past_marks, attendance))
    student_id = cur.lastrowid
    cur.execute("INSERT INTO predictions (student_id, predicted_score, risk_category) VALUES (?,?,?)",
                (student_id, prediction, risk))
    conn.commit()
    conn.close()

    return render_template('home.html', predicted=round(prediction,1), risk_level=risk, student=name)

@app.route('/dashboard')
def dashboard():
    conn = sqlite3.connect('education_system.db')
    cur = conn.cursor()
    cur.execute("""
        SELECT s.student_name, s.daily_study_hours, s.previous_marks, s.attendance_rate,
               p.predicted_score, p.risk_category
        FROM students s
        JOIN predictions p ON s.student_id = p.student_id
        ORDER BY p.prediction_time DESC
    """)
    records = cur.fetchall()
    conn.close()
    return render_template('dashboard.html', all_students=records)

@app.route('/export_data')
def export_data():
    conn = sqlite3.connect('education_system.db')
    df = pd.read_sql_query("""
        SELECT s.student_name, s.daily_study_hours, s.previous_marks, s.attendance_rate,
               p.predicted_score, p.risk_category, p.prediction_time
        FROM students s
        JOIN predictions p ON s.student_id = p.student_id
    """, conn)
    conn.close()
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv',
                     as_attachment=True, download_name='performance_report.csv')

@app.route('/upload_bulk', methods=['POST'])
def upload_bulk():
    file = request.files['csv_file']
    if not file:
        return "No file selected", 400

    df = pd.read_csv(file)
    required_cols = ['student_name', 'daily_study_hours', 'previous_marks', 'attendance_rate']
    if not all(col in df.columns for col in required_cols):
        return "CSV must have columns: student_name, daily_study_hours, previous_marks, attendance_rate", 400

    conn = sqlite3.connect('education_system.db')
    cur = conn.cursor()
    for _, row in df.iterrows():
        feats = np.array([[row['daily_study_hours'], row['previous_marks'], row['attendance_rate']]])
        pred_score = model.predict(feats)[0]
        pred_score = max(0, min(100, pred_score))
        risk_cat = classify_risk(pred_score)

        cur.execute("INSERT INTO students (student_name, daily_study_hours, previous_marks, attendance_rate) VALUES (?,?,?,?)",
                    (row['student_name'], row['daily_study_hours'], row['previous_marks'], row['attendance_rate']))
        sid = cur.lastrowid
        cur.execute("INSERT INTO predictions (student_id, predicted_score, risk_category) VALUES (?,?,?)",
                    (sid, pred_score, risk_cat))
    conn.commit()
    conn.close()
    return 'Bulk upload successful! <a href="/dashboard">View Dashboard</a>'

if __name__ == '__main__':
    app.run(debug=True)