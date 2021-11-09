import pandas as pd
import pickle 

with open('startup-model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('startup-ct.pickle', 'rb') as f:
    ct = pickle.load(f)
    
Xnew = pd.read_csv('new_company_ct.csv')
Xnew_org = Xnew
Xnew = ct.transform(Xnew)
Ynew = model.predict(Xnew)

for i in range (len(Ynew)):
    print(f'{Xnew_org.iloc[i]}\nVoitto: {Ynew[i][0]}\n')
    