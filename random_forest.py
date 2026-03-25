import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load your data
# Note: Ensure the filename matches what you saved previously
df = pd.read_csv('sample_data.csv', sep='\t')

# 2. Preprocess Categorical Data
# ML models need numbers. We'll convert text columns to numeric codes.
le = LabelEncoder()
df_encoded = df.copy()

categorical_cols = ['Creator', 'Source', 'Location', 'Day of Week']
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df[col])

# 3. Define Features (X) and Target (y)
X = df_encoded.drop('Label', axis=1)
y = df_encoded['Label']

# 4. Train the Model
# Random Forest is excellent for finding patterns in diverse data types.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Calculate and Display Feature Importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Property': X.columns,
    'Importance Score': importances
}).sort_values(by='Importance Score', ascending=False)

print("\n--- Property Influence Ranking ---")
print(feature_importance_df.to_string(index=False))
