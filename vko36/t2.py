# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7  2021

@author: ottol
"""

import pandas as pd

df_emp = pd.read_csv('employees.csv', header=0, dtype={'phone1':str, 'phone2':str})
df_dep = pd.read_csv('departments.csv', header=0)

emp_count = df.shape[0]