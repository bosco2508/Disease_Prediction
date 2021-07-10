import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import naive_bayes
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def test_cleaning(a):
    cols = a.columns
    data = a[cols].values.flatten()

    s = pd.Series(data)
    s = s.str.strip()
    s = s.values.reshape(a.shape)
    a = pd.DataFrame(s, columns=df.columns)
    a = a.fillna(0)
    return a

def convert_into_weights(a,b):
    vals = a.values
    symptoms = b['Symptom'].unique()
    for i in range(len(symptoms)):
        vals[vals == symptoms[i]] = b[b['Symptom'] == symptoms[i]]['weight'].values[0]
    cols= a.columns
    d = pd.DataFrame(vals, columns=cols)
    d = d.replace('dischromic _patches', 0)
    d = d.replace('spotting_ urination',0)
    new_df = d.replace('foul_smell_of urine',0)
    return new_df

def train_model(new_df):
    data = new_df.iloc[:,1:].values
    labels = new_df['Disease'].values
    data
    x_train, x_test, y_train, y_test = train_test_split(data, labels, shuffle=True, train_size = 0.85)
    model = SVC()

    model.fit(x_train, y_train)
    #Check Accuracy
    preds = model.predict(x_test)
    conf_mat = confusion_matrix(y_test, preds)
    df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
    print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)
    return model


df = pd.read_csv('./dataset2.csv')
df = test_cleaning(df)
df1 = pd.read_csv('./Symptom-severity_WithCorona.csv')
new_df = convert_into_weights(df,df1)
model= train_model(new_df)

#Predict Disease
df2= pd.read_csv('./Book1.csv')
df2 = test_cleaning(df2)
df2 = convert_into_weights(df2,df1) 
data2 = df2.iloc[:,1:].values
disease= model.predict(data2)
print(disease)

