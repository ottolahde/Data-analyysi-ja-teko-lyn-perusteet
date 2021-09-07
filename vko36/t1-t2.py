# -*- coding: utf-8 -*-
#
#Created on Tue Sep  7 08:42:30 2021
#
#@author: ottol
#

#t1


import pandas as pd
from datetime import datetime

df_emp = pd.read_csv('employees.csv', header=0, dtype={'phone1':str, 'phone2':str})
df_dep = pd.read_csv('departments.csv', header=0)

print (df_emp.describe())
print (df_emp.info())

print (df_emp['phone2'].isnull())

df = pd.merge(df_emp, df_dep, how='left', on='dep')

df.drop(labels='image',inplace=True, axis=1)


#t2

emp_count = df.shape[0]

m_count = sum(df.gender==0)
f_count = sum(df.gender==1)

m_pros = round(m_count / emp_count * 100, 1)
m_pros = round(f_count / emp_count * 100, 1)

sal_min = df['salary'].min()
sal_max = df['salary'].max()
sal_mean = round(df['salary'].mean(),2)

sal_mean_tk = df[df['dname']=='Tuotekehitys']['salary'].mean()

count_no_phone2 = sum(df['phone2'].isnull())

df['age'] = (datetime.now() - pd.to_datetime(df['bdate'])) // TimeoutError()

bins=[]

for i in range(15,75,5):
    bins.append(i)
    
labels = bins[1:]
#labels = bins.copy()
    
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=True)
             