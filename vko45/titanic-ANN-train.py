import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


df = pd.read_csv('titanic-class-age-gender-survived.csv')

# X = df.iloc[:, :-1]
X = df.iloc[:, [0,1,2]]
y = df.iloc[:, [-1]]


X_org = X
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['PClass','Gender'])], remainder='passthrough')
X = ct.fit_transform(X) # ensimmäisellä kerralla fit_transform

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)


# skaalataan X ja y
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)


# opetetaan malli
model = Sequential()
model.add(Dense(units=8, input_dim=X.shape[1],activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['mse'])
history=model.fit(X_train, y_train, epochs=100, batch_size=32, 
                  validation_data=(X_test, y_test))

#oppimisen visualisointi
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

y_pred_proba = model.predict(X_test)
y_pred = (model.predict(X_test) > 0.5)




cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
ps = precision_score(y_test, y_pred)
rc = recall_score(y_test, y_pred)

print (f'{cm}')
print (f'Mallin ulkoinen tarkkuus: {ac*100:.02f} %')
print (f'precision_score: {ps:.02f}')
print (f'recall_score: {rc:.02f}')

sns.heatmap(cm, annot=True, fmt='g')
plt.show()

# tn, fp, fn, tp = cm.ravel()

# df_new = pd.read_csv('titanic-new.csv')
# df_new_org = df_new
# df_new = df_new.iloc[:, [0,1,2]]
# df_new = ct.transform(df_new)

# y_new = model.predict(df_new)
# y_new_proba = model.predict_proba(df_new)

# for i in range (len(y_new)):
#     print (f'{df_new_org.iloc[i]}\nSelviytyminen: {y_new[i]} ({y_new_proba[i][1]:.02f})')
