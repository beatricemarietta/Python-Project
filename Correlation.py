import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/heart_clean.csv')

# Percentage of missing values
((df.isnull().sum()/df.shape[0])*100).sort_values(ascending=False)

#ANALYSIS
#Percentage
target=df['target'].replace(to_replace={1:'Disease', 0:'No Disease'})
target_valcount = target.value_counts()
target_labels = target_valcount.index
target_sizes = target_valcount.values

plt.figure(figsize=(8, 6))
plt.pie(target_sizes, labels=target_labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('Set2'))
plt.title('The percentage of Patients with Heart Disease dan No Disease')
plt.show()

#Correlation 
corr_df = df.corr()
f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(corr_df,annot=True,fmt=".2f",ax=ax,linewidths=0.5,linecolor="orange")
plt.title('Correlations between different predictors')
plt.show()
