# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 15:51:26 2021

@author: chenguimei
"""



# 加载模块
import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt; plt.style.use('classic')
# import matplotlib.pyplot as plt; plt.style.use('seaborn')


#波士顿房价数据集
from  sklearn.datasets import fetch_california_housing
boston=fetch_california_housing()
boston.data
boston.target
boston.feature_names
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)
boston_df['target'] = boston.target 

cols = [i for i in boston_df.columns[:-1]]

# 训练xgboost回归模型
model = xgb.XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=150)
model.fit(boston_df[cols], boston_df['target'].values)


# 获取feature importance
plt.figure(figsize=(15, 5))
plt.bar(range(len(cols)), model.feature_importances_)
plt.xticks(range(len(cols)), cols, rotation=-45, fontsize=14)
plt.title('Feature importance', fontsize=14)
# plt.show()

# model是在第1节中训练的模型
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(boston_df[cols])

#第一列是特征名称，第二列是特征的数值，第三列是各个特征在该样本中对应的SHAP值。
# 比如我们挑选数据集中的第30位
j = 30
player_explainer = pd.DataFrame()
player_explainer['feature'] = cols
player_explainer['feature_value'] = boston_df[cols].iloc[j].values
player_explainer['shap_value'] = shap_values[j]
player_explainer['base'] = model.predict(boston_df[cols]).mean() #就是预测的分数的均值
player_explainer['sum'] = player_explainer['shap_value'].sum() #特征的shap和
player_explainer['base+sum'] = player_explainer['base']+player_explainer['sum']
player_explainer
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[j], boston_df[cols].iloc[j])
shap.summary_plot(shap_values, boston_df[cols])
shap.summary_plot(shap_values, boston_df[cols], plot_type="bar")
#LSTAT位于最后一个，因此我们只需要提取最后一列
pd.DataFrame(shap_values).iloc[:,-1].apply(lambda x:abs(x)).mean()  #输出 3.7574333926117998
shap.dependence_plot('LSTAT', shap_values, boston_df[cols], interaction_index=None, show=False)
shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(boston_df[cols])
shap.summary_plot(shap_interaction_values, boston_df[cols], max_display=4)