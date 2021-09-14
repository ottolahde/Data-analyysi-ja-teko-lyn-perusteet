# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:02:57 2021

@author: ottol
"""

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('emp-dep.csv', dtype={'phone1':str, 'phone2':str})

#t1

plt.scatter(x=df['age'], y=df['salary'])
plt.show()

dep_countes = df['dname'].value_counts().sort_index()
dep_countes.plot(kind='bar')
plt.show()

dep_countes.plot(kind='barh')
plt.show()

# t2

agc = df['age_group'].value_counts()
agc.plot(kind='bar')

max_count = range(agc.max() + 1)

plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
plt.show()

#t3

gvc = df['gender'].value_counts()
gvc.plot(kind='pie', ylabel='', labels=['miehet', 'naiset'],
         startangle=270, autopct='%1.1f%%')
plt.show()

cag = df.groupby(['age_group', 'gender']).size().unstack()
fig, ax = plt.subplots()
ax = cag.plot(kind='bar')
ax.legend(['miehet','naiset'])
plt.title('Työntekijät ikäryhmittäin')
plt.ylabel('Lukumäärä')
plt.xlabel('Ikäryhmä')
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
plt.show()