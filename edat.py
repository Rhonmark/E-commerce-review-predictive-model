import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv('STD Cases.csv')

# print(df.shape)
# print(df)
# print(df.info())

# print(df)

# print(df[df['Population'].isna()])
# print(df[df['Rate per 100K'].isna()])

df = df.dropna(subset=['Year', 'Rate per 100K'])
# df = df.dropna(how='any')

x = df[['Year']]
y = df['Rate per 100K']

print(y.describe())

# print('Year: ', x)
# print('Rate per 100k: ', y)
# print(x.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
  x, y, test_size=0.2, 
  train_size=0.8, 
  random_state=42)

scaler = StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

model = LinearRegression()
model.fit(x,y)

y_pred = model.predict(x)

plt.scatter(x, y, alpha=0.6)
plt.plot(x, y_pred, color='red', label='Trend Line')
plt.xlabel('Year')
plt.ylabel('Rate per 100K')
plt.title('Testing')
plt.legend()
plt.grid(True)
plt.show()