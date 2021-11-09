import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = pd.Series([1,2,3,4,5,6,7,8])
y = 2 * x + 3

df = pd.DataFrame({'x':x, 'y':y})

df.plot(kind='scatter', x='x', y='y')
plt.show()

#X = df.loc[:, ['x','y']]
X = df.iloc[:, [0]]
Y = df.iloc[:, [1]]

regr = LinearRegression()
regr.fit(X, Y)

coef = regr.coef_
inter = regr.intercept_

print(f'Suoran yhtälö: y = {coef[0][0]}x + {inter[0]}')

y_pred = regr.predict([[5]])

df.plot(kind='scatter', x='x', y='y')
plt.scatter(5, y_pred, color='red')
plt.plot(df.x, df.y)
plt.show()



#df.plot( x='x', y='y' )
#df.plot(kind='scatter', x=5, y=y_pred)
#plt.show()