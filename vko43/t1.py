import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('iris.csv')

X = df.drop('species', axis=1)

#etsitään optimaalisin klustereiden määrä ('Elbow methdod')
wcss = []
for i in range (1, 11): # 1-10 klusterikandidaattia
    kmeans = KMeans(n_clusters = i, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 11),wcss)
plt.title('Elbow Method')
plt.xlabel('Klustereiden määrä')
plt.ylabel('Neliösumma')
plt.show

# päädytään kulstereiden määrään 3
model = KMeans(n_clusters=3, random_state=0)
model.fit(X)

# ennustetaan klusterit
y_pred = model.predict(X)

# lisätään klusterin X:n ja berrataan todellisiin lajeihin
X['C'] = y_pred
X.loc[X['C'] == 0, 'pre_s'] = 'versicolor'
X.loc[X['C'] == 1, 'pre_s'] = 'setosa'
X.loc[X['C'] == 2, 'pre_s'] = 'virginica'

# otetaan todellinen laji df:stä
X['real_s'] = df['species']

print()
print()