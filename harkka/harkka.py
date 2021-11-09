import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

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


#Ikä scatter
plt.line(x=df['Age'], y=df['Year'])
plt.show()


#Pie chart sukupuolesta
#gvc = df['Sex'].value_counts()
#gvc.plot(kind='pie', title='Sukupuoli vuosilta 1896-2016', labels=['miehet', 'naiset'],
#         startangle=270, autopct='%1.1f%%')


#old = Yold['Sex'].value_counts()
#old.plot(kind='pie', title='Sukupuoli vuosilta 1896-1920', labels=['miehet', 'naiset'],
#         startangle=270, autopct='%1.1f%%')


old = Ynew['Sex'].value_counts()
old.plot(kind='pie', title='Sukupuoli vuosilta 1996-2016', labels=['miehet', 'naiset'],
         startangle=270, autopct='%1.1f%%')