import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 1. Configuration
n_samples = 50
creators = ['Alice', 'Bob', 'Charlie', 'Delta_Bot']
sources = ['API', 'Manual_Entry', 'Web_Scraper']
locations = ['New York', 'London', 'Mumbai', 'Remote']
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
start_date = datetime(2023, 1, 1)

data = []

# 2. Generation Loop
for _ in range(n_samples):
    # Date/Time Logic
    random_days = random.randint(0, 364)
    random_seconds = random.randint(0, 86399)
    dt = start_date + timedelta(days=random_days, seconds=random_seconds)
    
    creator = random.choice(creators)
    source = random.choice(sources)
    location = random.choice(locations)
    duration = random.randint(10, 3600)
    day_of_week = days[dt.weekday()]
    time_hhmm = dt.strftime('%H%M')
    date_yyyymmdd = dt.strftime('%Y%m%d')
    
    # ML Logic (Score-based labeling)
    score = 0
    if source == 'API': score += 2
    if creator in ['Alice', 'Bob']: score += 1
    if 1000 <= int(time_hhmm) <= 1800: score += 1 # Daytime data is better
    if creator == 'Delta_Bot': score -= 3
    
    # Final Label with slight randomness
    label = 1 if (score + np.random.normal(0, 1)) > 0.5 else 0
    
    data.append([
        creator, source, duration, location, 
        day_of_week, time_hhmm, date_yyyymmdd, label
    ])

# 3. Create DataFrame and Save
columns = [
    'Creator', 'Source', 'Duration (sec)', 'Location', 
    'Day of Week', 'Time(HHMM)', 'Date (YYYYMMDD)', 'Label'
]
df = pd.DataFrame(data, columns=columns)

# Save to your project folder
#df.to_csv('sample_data.csv', index=False, sep='\t') # Saved as Tab-Separated to match your format
df_new = df.drop('Label', axis=1)
df_new.to_csv('test_data.csv', index=False, sep='\t') # Data generated for testing purposes
print("Dataset 'test_data.csv' created successfully.")

# Display first 5 rows
print(df_new.head())