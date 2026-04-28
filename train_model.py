import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

# Synthetic data
np.random.seed(42)
total_students = 500

hrs_studied = np.random.uniform(0, 12, total_students)
prev_scores = np.random.uniform(30, 100, total_students)
attend_percent = np.random.uniform(40, 100, total_students)

final_marks = (0.3 * hrs_studied + 0.5 * prev_scores + 0.2 * attend_percent + np.random.normal(0, 5, total_students))
final_marks = np.clip(final_marks, 0, 100)

data_df = pd.DataFrame({
    'study_hours': hrs_studied,
    'past_marks': prev_scores,
    'attendance': attend_percent,
    'final_marks': final_marks
})
data_df.to_csv('student_data.csv', index=False)

X = data_df[['study_hours', 'past_marks', 'attendance']]
y = data_df['final_marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
print(f"Linear Regression R²: {r2_score(y_test, lr_preds):.3f}")

# Random Forest
rf_model = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print(f"Random Forest R²: {r2_score(y_test, rf_preds):.3f}")

# Save best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("Model saved as 'best_model.pkl'")