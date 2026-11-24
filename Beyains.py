import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 
data = pd.read_csv('MachineLearningCVE\Monday-WorkingHours.pcap_ISCX.csv', header=0)

# 
X = data.iloc[:1000, :40]  # 
y = data.iloc[:1000, -1]   # 

# 
print("check...")
print(X.isin([np.inf, -np.inf]).sum())  # 
print(X.isnull().sum())  # 

# 
X.replace([np.inf, -np.inf], np.nan, inplace=True)  # 
X.fillna(X.median(), inplace=True)  # 

# 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 
model = BayesianRidge()
model.fit(X_train_scaled, y_train)

# 
weights = model.coef_

# 
feature_weights = pd.DataFrame({
    'Feature': X.columns,  # 
    'Weight': weights
})

# 
feature_weights = feature_weights.sort_values(by='Weight', ascending=False)

# 
top_20_features = feature_weights.head(20)

# 
print("Top 20 features by weight:")
print(top_20_features)


import pandas as pd

# 
feature_weights = pd.DataFrame({
    'Feature': ['Total Backward Packets', 'Total Length of Bwd Packets'],
    'Weight': [0.000002, 0.000002]
})

# 
feature_weights.to_csv('MachineLearningCVE\Monday-feature_weights.csv', index=False, header=True)

print("'feature_weights.csv'")


