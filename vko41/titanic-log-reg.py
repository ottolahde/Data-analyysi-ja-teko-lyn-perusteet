import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


df = pd.read_csv('titanic-class-age-gender-survived.csv')


# jaetaan X ja y
X = df.iloc[:, [0,1,2]]
y = df.iloc[:, [-1]]


# dummyt
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['PClass','Gender'])], remainder='passthrough')
X = ct.fit_transform(X) # ensimmäisellä kerralla fit_transform


# opetus ja testidata
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2,
random_state=0)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
pc = precision_score(y_test, y_pred)
rc = recall_score(y_test, y_pred)


print (f'cm: \n{cm}')
print (f'acc: {acc}')
print (f'pc: {pc}')
print (f'rc: {rc}')


sns.heatmap(cm, annot=True, fmt='g')
plt.show()

df_new = pd.read_csv('titanic-new.csv')
#df_new.drop('PClass', inplace=True, axis=1)

df_new_org = df_new
df_new = ct.transform(df_new)
y_new = model.predict(df_new)
y_new_proba = model.predict_proba(df_new)

for i in range (len(y_new)):
    print(f'\n{df_new_org.iloc[i]}\Selviytyminen: {y_new[i]} ({y_new_proba[i][1]:.2f})')