import pandas as pd

df_tit_data = pd.read_csv('Titanic_data.csv', header=0)
df_tit_names = pd.read_csv('Titanic_names.csv', header=0)

print (df_tit_data.describe())
print (df_tit_data.info())

print (df_tit_names.describe())
print (df_tit_names.info())

print(df_tit_data.hist(bins = 4))

df = pd.merge(df_tit_data, df_tit_names, how='inner', on='id')
