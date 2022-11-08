import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_columns=300
#
df = pd.read_csv(r'/Users/abs/Documents/Data Science/My-Projects/ANN-Final/lending_club_info.csv',index_col='LoanStatNew')
#
print(df.loc['revol_util']['Description'])

#
def feat_info(col_name):    print(df.loc[col_name]['Description'])

#
print(df.describe())
df = pd.read_csv(r'/Users/abs/Documents/Data Science/My-Projects/ANN-Final/lending_club_loan_two.csv')
print(df.describe())
sns.countplot(x='loan_status', data=df)
#
plt.figure(figsize=(12, 4))
sns.displot(df['loan_amnt'], bins=40, kde=False)
plt.xlim(0, 45000)
#
plt.figure(figsize=(12, 12))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.ylim(10.0)
#
feat_info('installment')
sns.scatterplot(x='installment', y='loan_amnt', data=df)
#

sns.boxplot(x='loan_status',y='loan_amnt',data=df)
#
df.groupby('loan_status')['loan_amnt'].describe()
 sorted(df['grade'].unique())
 sorted(df['sub_grade'].unique())

sns.countplot(x='grade',hue='loan_status',data= df)


plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data= df, subgrade_order=subgrade_order)

plt.figure(figsize=(12,4))
subgrade_order=sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,hue='loan_status',palette='coolwarm')

f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]
plt.figure(figsize=(12,4))
subgrade_order=sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data= f_and_g,order=subgrade_order,hue='loan_status')

df['loan_status'].unique()
df['loan_repaid']= df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
df[['loan_repaid','loan_status']]

plt.figure(figsize=(12,8))
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')

##missing data
len(df)
df.isnull().sum()
100* df.isnull().sum()/len(df)
feat_info('emp_title')
print('\n')
feat_info('emp_length')
df['emp_title'].nunique()
df['emp_title'].value_counts()
df = df.drop('emp_title',axis=1)

emp_length_order= sorted(df['emp_length'].dropna().unique())
plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df, order=emp_length_order)
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')

