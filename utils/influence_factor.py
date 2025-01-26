import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, f1_score, roc_auc_score, classification_report
from astropy.table import Table
from sklearn.metrics import roc_auc_score

df = pd.read_csv('student-data.csv')
dfv = pd.read_csv('student-data.csv')
output_dir = 'picture'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# mapping strings to numeric values:

df['school'] = df['school'].map({'GP': 0, 'MS': 1})
df['sex'] = df['sex'].map({'M': 0, 'F': 1})
df['address'] = df['address'].map({'U': 0, 'R': 1})
df['famsize'] = df['famsize'].map({'LE3': 0, 'GT3': 1})
df['Pstatus'] = df['Pstatus'].map({'T': 0, 'A': 1})
df['Mjob'] = df['Mjob'].map({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4})
df['Fjob'] = df['Fjob'].map({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4})
df['reason'] = df['reason'].map({'home': 0, 'reputation': 1, 'course': 2, 'other': 3})
df['guardian'] = df['guardian'].map({'mother': 0, 'father': 1, 'other': 2})
df['schoolsup'] = df['schoolsup'].map({'no': 0, 'yes': 1})
df['famsup'] = df['famsup'].map({'no': 0, 'yes': 1})
df['paid'] = df['paid'].map({'no': 0, 'yes': 1})
df['activities'] = df['activities'].map({'no': 0, 'yes': 1})
df['nursery'] = df['nursery'].map({'no': 0, 'yes': 1})
df['higher'] = df['higher'].map({'no': 0, 'yes': 1})
df['internet'] = df['internet'].map({'no': 0, 'yes': 1})
df['romantic'] = df['romantic'].map({'no': 0, 'yes' : 1})
df['passed'] = df['passed'].map({'no': 0, 'yes': 1})
# reorder dataframe columns :
col = df['passed']
# type(col) col的类型为df
del df['passed']
df['passed'] = col

    
# feature scaling will allow the algorithm to converge faster, large data will have same scal
#特征缩放操作，使得收敛速度更快，处于同一个数据维度
for i in df:
    col = df[i]
    # let's choose columns that have large values
    if(np.max(col)>6):
        Max = max(col)
        Min = min(col)
        mean = np.mean(col)
        col  = (col-mean)/(Max)
        df[i] = col
    elif(np.max(col)<6):
        col = (col-np.min(col))
        col /= np.max(col)
        df[i] = col

# for col in  [
#     "school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob",\
#     "Fjob","reason","guardian","traveltime","studytime","failures","schoolsup",\
#     "famsup","paid","activities","nursery","higher","internet","romantic","famrel",\
#     "freetime","goout","Dalc","Walc","health","absences","passedsex","age","address",\
#     "famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","guardian","traveltime",\
#     "studytime","failures","schoolsup","famsup","paid","activities","nursery","higher",\
#     "internet","romantic","famrel","freetime","goout","Dalc","Walc","health","absences","passed"]:
#     if col != 'passed': 
perc = (lambda col: col/col.sum())
index = [0,1]
alc_tab = pd.crosstab(index=df.passed, columns=dfv.Medu)
alc_perc = alc_tab.apply(perc).reindex(index)
alc_perc.plot.bar(colormap="Dark2_r", figsize=(14,6), fontsize=16)
plt.title(f'student status By {col}', fontsize=20)
plt.xlabel('Student status', fontsize=16)
plt.ylabel('Percentage of Student', fontsize=16)
plt.show()
