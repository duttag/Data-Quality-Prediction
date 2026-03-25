import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# 1. Load and Prepare Data
df = pd.read_csv('sample_data.csv', sep='\t')
df['Hour'] = df['Time(HHMM)'].apply(lambda x: int(str(x).zfill(4)[:2]))

# We'll focus on the most impactful features for the "Rules"
features = ['Creator', 'Source', 'Day of Week', 'Hour']
X = df[features].copy()
y = df['Label']

# 2. Encode Text to Numbers for the model
encoders = {}
for col in ['Creator', 'Source', 'Day of Week']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le  # Save for reference

# 3. Train a Simple Decision Tree (Max depth 3 to keep it readable)
clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
clf.fit(X, y)

# 4. Plot the Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, 
          feature_names=features, 
          class_names=['Bad', 'Good'], 
          filled=True, 
          rounded=True, 
          fontsize=12)

plt.title("Decision Tree: Rules for Data Quality", fontsize=20)
plt.show()

# 5. Print the encoding "key" so you know what the numbers mean
print("--- Encoding Key ---")
for col, le in encoders.items():
    print(f"{col}: {dict(zip(range(len(le.classes_)), le.classes_))}")
