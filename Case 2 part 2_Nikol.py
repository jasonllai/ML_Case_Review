#!/usr/bin/env python
# coding: utf-8

# %pip install tensorflow

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
get_ipython().system('pip install scikeras')
from sklearn.model_selection import GridSearchCV


# In[2]:


train_df = pd.read_csv('USCensusTraining.csv')


# In[3]:


train_df.isnull().sum().sort_values(ascending=False) # No missing value


# In[4]:


# Drop redundant attribute 'education'
train_df = train_df.drop(columns=['education'])


# In[5]:


X = train_df.drop(columns=['income'])
pd.DataFrame(X).describe()


# In[6]:


train_df['income'] = train_df['income'].str.replace('K', '', regex=False)
train_df['income'] = train_df['income'].str.replace('.', '', regex=False)
train_df['income'] = train_df['income'].str.strip()
y = train_df['income'].apply(lambda x: 1 if x == '>50' else 0)


# In[7]:


pd.DataFrame(y).describe()


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)


# In[9]:


# One-hot encode categorical variables
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
X_train = pd.get_dummies(X_train, columns=categorical_cols)
X_test_original = X_test.copy()
X_test = pd.get_dummies(X_test, columns=categorical_cols)


# In[10]:


X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# In[11]:


(X_train.columns == X_test.columns).all() 


# In[12]:


# Scale continuous variables
continuous_cols = ['age', 'demogweight', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
scaler = preprocessing.MinMaxScaler().fit(X_train[continuous_cols])
X_train[continuous_cols] = scaler.transform(X_train[continuous_cols])
X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])


# In[13]:


X_train.head()


# In[14]:


tf.random.set_seed(42)
model = Sequential(name = "ANN")
model.add(Input(shape = (X_train.shape[1], )))
model.add(Dense(32, activation = 'sigmoid', kernel_initializer = 'glorot_uniform', name = 'hidden'))
model.add(Dense(1, activation = 'sigmoid', name = 'output'))
model.compile(optimizer = RMSprop(learning_rate = 1e-3),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])


# In[15]:


history = model.fit(X_train, y_train, epochs = 40, batch_size = 32, validation_split = 0.2, verbose = 2)


# In[16]:


loss, acc = model.evaluate(X_test, y_test, verbose = 2)
print(f"Test accuracy: {acc: .3f}")


# ### b. Most Important Variables for Predicting Income

# In[17]:


X_mean = X_test.mean().to_numpy().reshape(1, -1)
X_min = X_test.min().to_numpy()
X_max = X_test.max().to_numpy()


# In[18]:


output_mean = model.predict(X_mean)
output_mean = np.array(output_mean).flatten()


# In[19]:


results = []
for i, col in enumerate(X_test.columns):
    x_low = X_mean.copy()
    x_high = X_mean.copy()

    x_low[0, i] = X_min[i]
    x_high[0, i] = X_max[i]

    y_low = np.array(model.predict(x_low)).flatten()
    y_high = np.array(model.predict(x_high)).flatten()

    change_low = np.abs(y_low - output_mean)
    change_high = np.abs(y_high - output_mean)

    mean_change = np.mean([change_low, change_high])

    results.append({
        'Feature': col,
        'Mean Change (Sensitivity)': mean_change
    })


# In[20]:


importance_df = pd.DataFrame(results).sort_values(
    'Mean Change (Sensitivity)', ascending=False
)

print(importance_df)


# In[21]:


importance_df['Original Feature'] = importance_df['Feature'].str.split('_').str[0]
importance_cat = (importance_df
                  .groupby('Original Feature')['Mean Change (Sensitivity)']
                  .mean()   # or .sum() if you prefer
                  .reset_index()
                  .sort_values('Mean Change (Sensitivity)', ascending=False))

print(importance_cat)


# Capital gain is the most important variables for predicting income since the greater the change in the output, the more important that feature is. 

# ### c. Explain the Predicted Accuracy

# The neural network correctly predicted whether income >50K or ≤50K for ~85.3% of the test data.

# ### d. Compare the predicted income with actual income. Which error is the model more prone to making? Is this type of error more protective of, say, banks or loan applicants?

# In[22]:


from sklearn.metrics import confusion_matrix

y_pred = (model.predict(X_test) > 0.5).astype(int)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, digits = 3))


# The recall (true positive rate) for class 1 (high income) is relatively low, indicating the model fails to correctly identify a substantial portion of high-income individuals. Many actual high-income people are misclassified as low-income.
# This type of error more protective for banks.

# ### e. Which occupations are associated with predicted income over $50,000?
# ### Which education levels? Which ages? Is this intuitive?

# In[23]:


y_pred = (model.predict(X_test) > 0.5).astype(int)
X_test_original['predicted_income'] = y_pred
high_income = X_test_original[X_test_original['predicted_income'] == 1]
occupation_counts = high_income['occupation'].value_counts()
print("Top occupations predicted >50K:\n", occupation_counts.head(10))


# In[24]:


education_counts = high_income['education-num'].value_counts()
print("\nTop education levels predicted >50K:\n", education_counts.head(10))


# In[25]:


mean_age = high_income['age'].mean()
median_age = high_income['age'].median()
print(f"\nMean age for predicted >50K: {mean_age:.1f}")
print(f"Median age for predicted >50K: {median_age:.1f}")


# Exec-managerial and Prof-specialty are associated with predicted income over 50,000. 
# Education level that is 13 yrs is associated with predicted income over 50,000. 
# Age around 44-45 is associated with predicted income over 50,000. 

# ### Construct graphs of the top three categorical predictors, and their relationship to predicted income

# In[26]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[27]:


top_categorical = ['marital-status', 'relationship', 'occupation']
fig, axes = plt.subplots(1, 3, figsize=(20,6))

for i, feature in enumerate(top_categorical):

    prop_df = (X_test_original
               .groupby(feature)['predicted_income']
               .mean()
               .sort_values(ascending=False)
               .reset_index())

    sns.barplot(x=feature, y='predicted_income', data=prop_df, ax=axes[i], palette='viridis')

    axes[i].set_ylabel('Proportion >50K')
    axes[i].set_xlabel(feature)
    axes[i].set_title(f'{feature} vs Predicted Income')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# ### f. Construct a histogram of one numeric variable which is important in the model, with an overlay of income

# In[28]:


X_test_original['capital-gain'].describe()


# In[29]:


X_test_original['actual_income'] = y_test.values
bins = [0, 1000, 5000, 10000, 20000, 50000, X_test_original['capital-gain'].max()]
labels = ['0–1k', '1k–5k', '5k–10k', '10k–20k', '20k–50k', '>50k']
X_test_original['capital_gain_bin'] = pd.cut(
    X_test_original['capital-gain'], bins=bins, labels=labels, include_lowest=True
)

capital_crosstab = pd.crosstab(
    X_test_original['capital_gain_bin'],
    X_test_original['actual_income'],
    normalize='index'
)

capital_crosstab.plot(
    kind='bar',
    stacked=True,
    figsize=(8,5),
    colormap='viridis'
)
plt.ylabel('Proportion of Actual Income')
plt.xlabel('Capital Gain')
plt.title('Capital Gain vs Actual Income')
plt.legend(title='Income', labels=['<=50K', '>50K'])
plt.xticks(rotation=45)
plt.show()


# In[30]:


X_test_original['demogweight'].describe()


# In[31]:


X_test_original['demogweight_bin'] = pd.qcut(X_test_original['demogweight'], q=6)
capital_crosstab = pd.crosstab(
    X_test_original['demogweight_bin'],
    X_test_original['actual_income'],
    normalize='index'
)


capital_crosstab.plot(kind='bar', stacked=True, figsize=(8,5), colormap='viridis')
plt.ylabel('Proportion of Actual Income')
plt.xlabel('Demogweight')
plt.title('Demogweight vs Actual Income')
plt.legend(title='Actual Income', labels=['<=50K', '>50K'])
plt.xticks(rotation=45)
plt.show()


# # Part 2: Tuning

# ## Training set imputation for ? values

# In[32]:


(train_df == '?').sum()


# ### Imputation

# In[33]:


categorical_cols_missing = ['workclass', 'occupation', 'native-country']

for c in categorical_cols_missing:
    train_df[c] = train_df[c].replace('?', np.nan)

for c in categorical_cols_missing:
    for i in range(len(train_df)):
        if pd.isna(train_df.loc[i, c]):
            same_rows = train_df[
                (train_df['sex'] == train_df.loc[i, 'sex']) &
                (train_df['marital-status'] == train_df.loc[i, 'marital-status']) &
                (~train_df[c].isna())
            ]
            if not same_rows.empty:
                mode_val = same_rows[c].mode()[0]
                train_df.loc[i, c] = mode_val
            else:
                train_df.loc[i, c] = train_df[c].mode()[0]


# ### Training the model again --> Test Accuracy Didn't Change

# In[34]:


tf.random.set_seed(42)
model = Sequential(name = "ANN")
model.add(Input(shape = (X_train.shape[1], )))
model.add(Dense(32, activation = 'sigmoid', kernel_initializer = 'glorot_uniform', name = 'hidden'))
model.add(Dense(1, activation = 'sigmoid', name = 'output'))
model.compile(optimizer = RMSprop(learning_rate = 1e-3),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs = 40, batch_size = 32, validation_split = 0.2, verbose = 2)


loss, acc = model.evaluate(X_test, y_test, verbose = 2)
print(f"Test accuracy: {acc: .3f}")


# ### Tuning Parameters

# In[35]:


def build_model(meta,optimizer="RMSprop", lr=1e-3,units=32,units2=0, act="sigmoid", momentum=0.9, nesterov=False):
    model = Sequential(name="ANN_Tunable")
    model.add(tf.keras.Input(shape=(meta["n_features_in_"],), name="input_features"))
    model.add(Dense(units, activation=act, kernel_initializer="glorot_uniform", name="hidden"))
    if units2 > 0:
        model.add(Dense(units2, activation=act, kernel_initializer="glorot_uniform", name="hidden2"))
    model.add(Dense(1, activation="sigmoid", name="output"))

    if isinstance(optimizer,str):
        opt_name = optimizer.lower()
        if opt_name == "rmsprop":
            opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
        elif opt_name in ("sgd", "sgd_m"):
            opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
        else:
            opt = tf.keras.optimizers.get(optimizer)

    else:
        opt = optimizer

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


# In[36]:


from scikeras.wrappers import KerasClassifier
clf = KerasClassifier(model=build_model, epochs= 40, batch_size=32, verbose=0)
param_grid = {
    "model__units": [16,32,64],
    "model__units2":  [0, 32, 64],
    "model__optimizer": ["RMSprop", "sgd"],
    "model__lr": [1e-2,1e-3],
    "model__momentum": [0.0, 0.5, 0.9, 0.95],
    "model__nesterov": [False, True]
}


# In[37]:


grid = GridSearchCV(clf, param_grid=param_grid, cv=3, scoring="accuracy", n_jobs=-1)


# In[38]:


grid.fit(X_train, y_train)


# In[39]:


print("Best params:", grid.best_params_)
print("Best CV acc.:", grid.best_score_)

