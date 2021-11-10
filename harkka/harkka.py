import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


df = pd.read_csv('athlete_events.csv')

print (df.describe())
#print(df.info())

#Dropattu ID
X = df.drop('ID', axis=1)

#Sortattu vuoden mukaan ja reset index
Y = df.sort_values('Year')
Y = Y.reset_index(drop=True)

Yold = Y.drop(df.index[16782:])
Ynew = Y.drop(df.index[:168472])

#Kaikesta vilkaisu
X.hist()
plt.show()



#Pie chart sukupuolesta
gvc = df['Sex'].value_counts()
gvc.plot(kind='pie', title='Sukupuoli vuosilta 1896-2016', labels=['miehet', 'naiset'],
        startangle=270, autopct='%1.1f%%')


old = Yold['Sex'].value_counts()
old.plot(kind='pie', title='Sukupuoli vuosilta 1896-1920', labels=['miehet', 'naiset'],
        startangle=270, autopct='%1.1f%%')


old = Ynew['Sex'].value_counts()
old.plot(kind='pie', title='Sukupuoli vuosilta 1996-2016', labels=['miehet', 'naiset'],
          startangle=270, autopct='%1.1f%%')


#Ikäjakauma
plt.figure(figsize=(12,6))
plt.title("Ikäjakauma")
plt.xlabel('Age')
plt.ylabel('Number of participants')
plt.hist(df.Age, bins = np.arange(10,80,2), color='purple' , edgecolor = 'white')

# Maakohtainen osallistumismäärä plot
top_10_countries = df.Team.value_counts().sort_values(ascending=False).head(10)
plt.figure(figsize=(12,6))
plt.title('Maakohtainen osallistumismäärä')
sns.barplot(x=top_10_countries.index, y=top_10_countries, palette = 'rocket')

#Mitallit
not_null_medals = df[(df['Height'].notnull()) & (df['Weight'].notnull())]
plt.figure(figsize =(12,6))
axis = sns.scatterplot(x = 'Height', y = 'Weight', data = not_null_medals, hue = 'Sex')
plt.title('Pituus ja paino Olympiamitalisteilla')