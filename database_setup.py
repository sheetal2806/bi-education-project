import sqlite3

def create_tables():
    conn = sqlite3.connect('education_system.db')
    cursor = conn.cursor()

    # Students table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        student_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_name TEXT NOT NULL,
        daily_study_hours REAL,
        previous_marks REAL,
        attendance_rate REAL
    )
    """)

    # Predictions table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        pred_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        predicted_score REAL,
        risk_category TEXT,
        prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (student_id) REFERENCES students(student_id)
    )
    """)

    conn.commit()
    conn.close()
    print("Database 'education_system.db' created successfully.")

if __name__ == '__main__':
    create_tables()