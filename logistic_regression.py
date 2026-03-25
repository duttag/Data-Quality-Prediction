import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# 1. Load the data
df = pd.read_csv('sample_data.csv', sep='\t')

# 2. Feature Engineering: Convert Time(HHMM) to a simple Hour (0-23)
df['Hour'] = df['Time(HHMM)'].apply(lambda x: int(str(x).zfill(4)[:2]))

# Select features for the model
features = ['Creator', 'Source', 'Duration (sec)', 'Location', 'Day of Week', 'Hour']
X = df[features].copy()
y = df['Label']

# 3. Preprocessing: Convert text to numbers
le = LabelEncoder()
for col in ['Creator', 'Source', 'Location', 'Day of Week']:
    X[col] = le.fit_transform(X[col])

# 4. Split Data: 120 for training, 30 for testing
# shuffle=False ensures we take the first 120 and the last 30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=30, train_size=120, shuffle=False)

# 5. Scale features (Logistic Regression performs better when data is normalized)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 7. Test the Model
y_pred = model.predict(X_test_scaled)

# 8. Print Results
print("--- Logistic Regression Test Results (N=30) ---")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Save the model and the scaler after training
joblib.dump(model, 'logistic_model.joblib')
joblib.dump(scaler, 'data_scaler.joblib')
print("Model and Scaler saved to disk.")
