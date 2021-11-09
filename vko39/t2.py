import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

df = pd.read_csv('salary.csv')

df.plot(kind='scatter', x='YearsExperience', y='Salary')
plt.show()

X = df.iloc[:, [0]]
y = df.iloc[:, [1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

regr = LinearRegression()
regr.fit(X_train, y_train)

coef = regr.coef_
inter = regr.intercept_

print(f'Suoran yhtälö on: y = {coef}x + {inter}')

y_pred = regr.predict(X_test)

r2 = r2_score(y_test, y_pred)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print (f'r2: {r2}')
print (f'mae: {mae}')
print (f'mse: {mse}')
print (f'rmse: {rmse}')

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_test, y_pred, color ='blue')
plt.show()

print(f'Uuden työntekijän palkka 7v kokemuksella on: {regr.predict([[7]])}')
