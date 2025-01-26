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

df = pd.read_csv('dataset/dadaset_OULAD/studentInfo_AAA.csv')
# columns_to_display = ['higher', 'Medu', 'Fedu', 'paid', 'failures', 'goout', 'age', 'schoolsup', 'passed']
# selected_data = df[columns_to_display]

# # 保存为新的 CSV 文件
# output_path = 'D:/selected_columns.csv'
# selected_data.to_csv(output_path, index=False)
passed_counts = df['final_result'].value_counts()
print(passed_counts)
# print(f"Selected columns saved to {output_path}")