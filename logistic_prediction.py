import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# 1. Load the persistent model and scaler
model = joblib.load('logistic_model.joblib')
scaler = joblib.load('data_scaler.joblib')

# 2. Read the unlabeled data
# Assuming test_data.csv has the same columns as your training data minus the label
df_new = pd.read_csv('test_data.csv', sep='\t')

# 3. Preprocess the new data
# Extract Hour from Time(HHMM) just like before
df_new['Hour'] = df_new['Time(HHMM)'].apply(lambda x: int(str(x).zfill(4)[:2]))

# We need to encode the text columns. 
# NOTE: In a real production app, you should also save/load your LabelEncoders
le = LabelEncoder()
features_to_encode = ['Creator', 'Source', 'Location', 'Day of Week']
X_new = df_new[['Creator', 'Source', 'Duration (sec)', 'Location', 'Day of Week', 'Hour']].copy()

for col in features_to_encode:
    X_new[col] = le.fit_transform(df_new[col])

# 4. Scale the features using the SAVED scaler
X_new_scaled = scaler.transform(X_new)

# 5. Predict
predictions = model.predict(X_new_scaled)

# 6. Append the results back to the original dataframe
df_new['Predicted_Label'] = predictions
df_new['Status'] = df_new['Predicted_Label'].map({1: 'Good', 0: 'Bad'})

# 7. Returns an array: [prob_of_0, prob_of_1]
probs = model.predict_proba(X_new_scaled)

# 8. Extract the probability of the predicted class as the 'Confidence'
df_new['Confidence_Score'] = probs.max(axis=1).round(2)  # Round to 2 decimal places for readability

# 9. (Optional) Extract specifically the 'Good' probability
df_new['Good_Probability'] = probs[:, 1].round(2)  # Round to 2 decimal places for readability

# 10. Save the results
df_new.to_csv('predicted_results.csv', index=False)
print("Predictions complete. Results saved to 'predicted_results.csv'.")
print(df_new[['Creator', 'Time(HHMM)', 'Status']].head())
