#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import make_column_transformer, ColumnTransformer
import statsmodels.api as sm
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
sns.set()


# In[2]:


data = pd.read_csv('/Users/priti16/Downloads/practical_application_II_starter-2/data/vehicles.csv')
data.head()


# In[3]:


df = data.copy()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe().round(2)


# In[7]:


df.describe(include='object')


# In[8]:


df['price'].unique()


# In[9]:


df['price'] = [0 if i=='-' else int(i) for i in df['price']]


# In[10]:


df['price'].dtype


# In[11]:


df['manufacturer'].unique()


# In[12]:


df['manufacturer'] = [float(i.split()[0]) for i in df['manufacturer']]


# In[ ]:


df['manufacturer'].dtype


# In[ ]:


df['odometer'].unique()


# In[ ]:


df['odometer'] = [float(i.split()[0]) for i in df['odometer']]


# In[ ]:


df['odometer'].dtype


# In[ ]:


df.head()


# In[ ]:


df.drop('title_status',axis=1, inplace=True)


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# In[ ]:


df.dropna()


# In[ ]:


df.isna().sum()


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.dropna(inplace=True)


# In[ ]:


df.isna().sum()


# In[ ]:


df.drop('id',axis=1,inplace=True) # Here ID column was dropped because there is no importance of the column


# In[ ]:


df = df.reset_index(drop=True)


# In[ ]:


df


# In[ ]:


numerical_data = df.select_dtypes(include='number')
numerical_data.head()


# In[ ]:


categorical_data = df.select_dtypes(include='object')
categorical_data.head()


# In[ ]:


fig, axes = plt.subplots(3,2, figsize=(10, 10))

axes[0,0].hist(numerical_data['price'])
axes[0,1].hist(numerical_data['year'])
axes[1,0].hist(numerical_data['odometer'])


axes[0, 0].set_title('price')
axes[0, 1].set_title('year')
axes[1, 0].set_title('odometer')


plt.suptitle('Histograms of Numerical Features', fontsize = 16)
plt.tight_layout()
plt.show()


# In[ ]:


fig, axes = plt.subplots(3,2, figsize=(10, 10))

sns.kdeplot(numerical_data['price'], fill=True, ax=axes[0,0])
sns.kdeplot(numerical_data['year'], fill=True, ax=axes[0,1])
sns.kdeplot(numerical_data['odometer'], fill=True, ax=axes[1,0])

fig.suptitle('Distribution of Numerical Features', fontsize=16)
plt.tight_layout()
plt.show()


# In[ ]:


fig, axes = plt.subplots(3,2, figsize=(10, 10))

axes[0,0].boxplot(numerical_data['price'])
axes[0,1].boxplot(numerical_data['year'])
axes[1,0].boxplot(numerical_data['odometer'])


axes[0, 0].set_title('price')
axes[0, 1].set_title('year')
axes[1, 0].set_title('odometer')

plt.suptitle('Boxplots of Numerical Features', fontsize=16)
plt.tight_layout()
plt.show()


# In[ ]:


price_selected = numerical_data[numerical_data['price']<100000]


# In[ ]:


len(price_selected)


# In[ ]:


price_selected = price_selected.reset_index()


# In[ ]:


price_selected


# In[ ]:


plt.hist(price_selected['price'])
plt.show()


# In[ ]:


plt.boxplot(price_selected['price'])
plt.show()


# In[ ]:


sns.kdeplot(np.sqrt(price_selected['price']), fill=True)
plt.show()


# In[ ]:


class IQR:
    def __init__(self, feature, data):
        self.feature = feature
        self.data = data

    def calculate_iqr(self):
        q1 = np.percentile(self.data[self.feature], 25)
        q3 = np.percentile(self.data[self.feature], 75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr
        self.outliers = self.data[((self.data[self.feature] < lower_limit) | (self.data[self.feature] > upper_limit))]
        return  {
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_limit': lower_limit,
            'upper_limit': upper_limit
        }


# In[ ]:


numerical_data.columns


# In[ ]:


year_iqr = IQR('year', price_selected)


# In[ ]:


year_iqr.calculate_iqr()


# In[ ]:


year_iqr.outliers


# In[ ]:


year_iqr.outliers['price'].values


# In[ ]:


year_iqr.outliers.index


# In[ ]:


year_selected = price_selected.iloc[~price_selected.index.isin(year_iqr.outliers.index)]


# In[ ]:


year_selected


# In[ ]:


len(year_selected)


# In[ ]:


plt.hist(year_selected['year'])
plt.show()


# In[ ]:


plt.boxplot(year_selected['year'])
plt.show()


# In[ ]:


numerical_data.columns


# In[ ]:


odometer_iqr = IQR('odometer', year_selected)


# In[ ]:


odometer_iqr.calculate_iqr()


# In[ ]:


odometer_iqr.outliers


# In[ ]:


odometer_iqr.outliers.describe()['odometer']


# In[ ]:


odometer_selected = year_selected[year_selected['odometer']<600000]


# In[ ]:


odometer_selected


# In[ ]:


plt.hist(odometer_selected['odometer'])
plt.show()


# In[ ]:


plt.boxplot(odometer_selected['odometer'])
plt.show()


# In[ ]:


fig, axes = plt.subplots(3,2, figsize=(10, 10))

sns.kdeplot(odometer_selected['price'], fill=True, ax=axes[0,0])
sns.kdeplot(odometer_selected['year'], fill=True, ax=axes[0,1])
sns.kdeplot(odometer_selected['odometer'], fill=True, ax=axes[1,0])


fig.suptitle('Distribution of Numerical Features in Final Dataset', fontsize=16)
plt.tight_layout()
plt.show()


# In[ ]:


fig, axes = plt.subplots(3,2, figsize=(10, 10))

axes[0,0].boxplot(odometer_selected['price'])
axes[0,1].boxplot(odometer_selected['year'])
axes[1,0].boxplot(odometer_selected['odometer'])

axes[0, 0].set_title('price')
axes[0, 1].set_title('year')
axes[1, 0].set_title('odometer')

plt.suptitle('Boxplots of Numerical Features in Final Dataset', fontsize=16)
plt.tight_layout()
plt.show()


# In[ ]:


numerical_selected = odometer_selected.reset_index(drop=True)


# In[ ]:


numerical_selected


# In[ ]:


numerical_selected['index'].values


# In[ ]:


categorical_data = categorical_data.iloc[numerical_selected['index'].values]


# In[ ]:


categorical_data = categorical_data.reset_index(drop=True)


# In[ ]:


categorical_data


# In[ ]:


df2 = pd.concat([numerical_selected, categorical_data], axis=1)
df2.head()


# In[ ]:


df2.shape


# In[ ]:


categorical_data


# In[ ]:


fig = plt.figure(figsize=(10,20))

ax1 = plt.subplot(5,2,1)
categorical_data['manufacturer'].value_counts().to_frame().plot(kind = 'bar', ax=ax1)

ax2 = plt.subplot(5,2,2)
categorical_data['model'].value_counts().to_frame().plot(kind = 'bar', ax=ax2)

ax3 = plt.subplot(5,2,3)
categorical_data['condition'].value_counts().to_frame().plot(kind = 'bar', ax=ax3)

plt.tight_layout()
plt.show()


# In[ ]:


df2_categorical = list(df2.columns[df2.dtypes=='object'])
df2_categorical


# In[ ]:


fig = plt.figure(figsize=(10,20))

for i in df2_categorical:
    ax = plt.subplot(5,2,df2_categorical.index(i)+10)
    df2.pivot_table(values='price', index=i, aggfunc='mean').sort_values(by='price').plot(kind='bar', ax=ax)
    plt.title('Average Price per {}'.format(i))
plt.tight_layout()
plt.show()


# In[ ]:


df2_numerical = list(df2.columns[df2.dtypes!='object'])
df2_numerical


# In[ ]:


fig = plt.figure(figsize=(10,20))

for i in df2_numerical:
    ax = plt.subplot(4,2,df2_numerical.index(i)+8)
    sns.scatterplot(x=df2[i], y=df2['price'], ax=ax)
    
plt.tight_layout()
plt.show()


# In[ ]:


df2.head()


# In[ ]:


X = df2.iloc[:,1:]
y = df2['price']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)


# In[ ]:


sns.pairplot(X_train)
plt.show()


# In[ ]:


corr_matrix = df2.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool),k=1)
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix,annot=True,cmap='RdBu',vmin=-1,vmax=1,mask=mask)
plt.show()


# In[ ]:


x_numeric_train = X_train.select_dtypes('number')
x = sm.add_constant(x_numeric_train)
results = sm.OLS(y_train,x).fit()
results.summary()


# In[ ]:


x_numeric_train


# In[ ]:


mutual_info = mutual_info_regression(x_numeric_train, y_train)
mutual_info = pd.Series(mutual_info)
mutual_info.index = x_numeric_train.columns
mutual_info.sort_values(ascending=False)


# In[ ]:


mutual_info.sort_values(ascending=False).plot(kind='bar')
plt.show()


# In[ ]:


X_train.drop('odometer', axis=1, inplace=True)
X_test.drop('odometer', axis=1, inplace=True)


# In[ ]:


x_numeric_train = X_train.select_dtypes('number')
x = sm.add_constant(x_numeric_train)
results = sm.OLS(y_train,x).fit()
results.summary()


# In[ ]:


categorical_preprocessor = Pipeline(
    steps = [('ohe', OneHotEncoder(drop='first'))]
)


# In[ ]:


numerical_preprocessor = Pipeline(
    steps = [('minmaxscaler' , MinMaxScaler())]
)


# In[ ]:


preprocessor = ColumnTransformer(
    [('odometer', categorical_preprocessor, list(X_train.select_dtypes('object').columns)),
    ('numerical', numerical_preprocessor, list(X_train.select_dtypes('number').columns))]   
)


# In[ ]:


preprocessor


# In[ ]:


OneHotEncoder(handle_unknown='ignore')


# In[ ]:


OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)


# In[ ]:


X_train_preproccessed = preprocessor.fit_transform(X_train)
X_test_preproccessed = preprocessor.transform(X_test)


# In[ ]:


OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)


# In[ ]:


X_train_preproccessed.shape


# In[ ]:


cval = KFold(n_splits=5, shuffle=True, random_state=99)


# In[ ]:


models = []
avg_errors = []


# In[ ]:


lr = LinearRegression()


# In[ ]:


errors_lr = -cross_val_score(estimator=lr,                  
                X=X_train_preproccessed,
                y=y_train,
                cv=cval,
                scoring='neg_root_mean_squared_error')

print('Errors {}'.format(errors_lr))
print()
print('Avg_error {}'.format(np.mean(errors_lr)))

avg_errors.append(np.mean(errors_lr))
models.append('Linear Regression')


# In[ ]:


neighbors = []
errors = []

for i in range(1, 20):
    knn = KNeighborsRegressor(n_neighbors=i)
    model = knn.fit(X_train_preproccessed, y_train)
    error = np.mean(-cross_val_score(estimator=model,                  
                        X=X_train_preproccessed,
                        y=y_train,
                        cv=cval,
                        scoring='neg_root_mean_squared_error'))
    neighbors.append(i)
    errors.append(error)


# In[ ]:


plt.plot(neighbors, errors)
plt.show()


# In[ ]:


np.argmin(errors)+1


# In[ ]:


knn = KNeighborsRegressor(n_neighbors=4)


# In[ ]:


errors_knn = -cross_val_score(estimator=knn,                  
                 X=X_train_preproccessed,
                 y=y_train,
                 cv=5,
                 scoring='neg_root_mean_squared_error')

print('Errors {}'.format(errors_knn))
print()
print('Avg_error {}'.format(np.mean(errors_knn)))

avg_errors.append(np.mean(errors_knn))
models.append('KNeighbors') 


# In[ ]:


dt = DecisionTreeRegressor(random_state=99)


# In[ ]:


errors_dt = -cross_val_score(estimator=dt,                  
                 X=X_train_preproccessed,
                 y=y_train,
                 cv=cval,
                 scoring='neg_root_mean_squared_error')

print('Errors {}'.format(errors_dt))
print()
print('Avg_error {}'.format(np.mean(errors_dt)))

avg_errors.append(np.mean(errors_dt))
models.append('Decision Tree')  


# In[ ]:


rf = RandomForestRegressor(random_state=99)


# In[ ]:


errors_rf = -cross_val_score(estimator=rf,                  
                 X=X_train_preproccessed,
                 y=y_train,
                 cv=cval,
                 scoring='neg_root_mean_squared_error')

print('Errors {}'.format(errors_rf))
print()
print('Avg_error {}'.format(np.mean(errors_rf)))

avg_errors.append(np.mean(errors_rf))
models.append('Random Forest')  


# In[ ]:


svr = SVR()


# In[ ]:


errors_svr = -cross_val_score(estimator=svr,                  
                 X=X_train_preproccessed,
                 y=y_train,
                 cv=cval,
                 scoring='neg_root_mean_squared_error')

print('Errors {}'.format(errors_svr))
print()
print('Avg_error {}'.format(np.mean(errors_svr)))

avg_errors.append(np.mean(errors_svr))
models.append('SVR')  


# In[ ]:


compare_models = pd.DataFrame({'Model':models, 'MSE':avg_errors}).sort_values(by='MSE', ascending=True)
compare_models


# In[ ]:


compare_models.plot(kind = 'bar')
plt.xlabel('Model')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Mean Squared Error for Different Models')
plt.xticks(range(len(compare_models)) ,compare_models['Model'] ,rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.tight_layout()
plt.show()


# In[ ]:




