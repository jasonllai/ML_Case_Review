#!/usr/bin/env python
# coding: utf-8

# ## Part 2 from the beginning

# In[1]:


import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
get_ipython().system('pip install scikeras')
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score


# In[2]:


train_df = pd.read_csv('USCensusTraining.csv')


# ### Replacing ? values

# In[3]:


train_df['income'] = train_df['income'].str.replace('K', '', regex=False)
train_df['income'] = train_df['income'].str.replace('.', '', regex=False)
train_df['income'] = train_df['income'].str.strip()

y = train_df['income'].apply(lambda x: 1 if x == '>50' else 0)

if 'education' in train_df.columns:
    train_df = train_df.drop(columns=['education'])

X = train_df.drop(columns=['income'])


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# In[5]:


categorical_cols_missing = ['workclass', 'occupation', 'native-country']

# Source: ChatGPT (GPT-5, OpenAI, Oct 2025) - code suggestion for imputing missing variables
for c in categorical_cols_missing:
    X_train[c] = X_train[c].replace('?', np.nan)
    X_test[c]  = X_test[c].replace('?', np.nan)

for c in categorical_cols_missing:
    for idx in X_train.index:
        if pd.isna(X_train.loc[idx, c]):
            mask = (
                (X_train['sex'] == X_train.loc[idx, 'sex']) &
                (X_train['marital-status'] == X_train.loc[idx, 'marital-status']) &
                (~X_train[c].isna())
            )
            same_rows = X_train.loc[mask, c]
            if not same_rows.empty:
                X_train.loc[idx, c] = same_rows.mode().iloc[0]
            else:
                mode_list = X_train[c].dropna().mode()
                if not mode_list.empty:
                    X_train.loc[idx, c] = mode_list.iloc[0]


    for idx in X_test.index:
        if pd.isna(X_test.loc[idx, c]):
            mask = (
                (X_train['sex'] == X_test.loc[idx, 'sex']) &
                (X_train['marital-status'] == X_test.loc[idx, 'marital-status']) &
                (~X_train[c].isna())
            )
            same_rows = X_train.loc[mask, c]
            if not same_rows.empty:
                X_test.loc[idx, c] = same_rows.mode().iloc[0]
            else:
                mode_list = X_train[c].dropna().mode()
                if not mode_list.empty:
                    X_test.loc[idx, c] = mode_list.iloc[0]


# In[6]:


categorical_cols = ['workclass','marital-status','occupation','relationship','race','sex','native-country']
categorical_cols = [c for c in categorical_cols if c in X_train.columns]

X_train = pd.get_dummies(X_train, columns=categorical_cols)
X_test  = pd.get_dummies(X_test,  columns=categorical_cols)
X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)

continuous_cols = ['age','demogweight','education-num','capital-gain','capital-loss','hours-per-week']
continuous_cols = [c for c in continuous_cols if c in X_train.columns]

scaler = MinMaxScaler().fit(X_train[continuous_cols])
X_train.loc[:, continuous_cols] = scaler.transform(X_train[continuous_cols])
X_test.loc[:,  continuous_cols] = scaler.transform(X_test[continuous_cols])


# In[7]:


tf.random.set_seed(42)
model = Sequential(name="ANN")
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(32, activation='sigmoid', kernel_initializer='glorot_uniform', name='hidden'))
model.add(Dense(1, activation='sigmoid', name='output'))
model.compile(optimizer=RMSprop(learning_rate=1e-3),loss='binary_crossentropy',metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train,epochs=40, batch_size=32, validation_split=0.2, callbacks=[es],verbose=2)

train_acc = history.history['accuracy'][-1]
val_acc   = history.history['val_accuracy'][-1]
print(f"Final train acc: {train_acc:.3f}  |  Final val acc: {val_acc:.3f}")

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.3f}")


# ## Parameter Tuning

# In[8]:


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


# In[9]:

# Source: ChatGPT (GPT-5, OpenAI, Oct 2025) - code suggestion for adding early stopping to prevent overfitting
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
clf = KerasClassifier(model=build_model, epochs= 40, batch_size=32, verbose=0)
param_grid = {
    "model__units": [16,32,64],
    "model__units2":  [0, 32, 64],
    "model__optimizer": ["RMSprop", "sgd"],
    "model__lr": [1e-2,1e-3],
    "model__momentum": [0.0, 0.5, 0.9, 0.95],
    "model__nesterov": [False, True]
}


# In[10]:


grid = GridSearchCV(clf, param_grid=param_grid, cv=3, scoring="accuracy", n_jobs=-1)


# In[11]:


grid.fit(X_train, y_train)


# ## Best Parameters

# In[15]:


print("Best parameters:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)


# ## Test Accuracy with the tuned model

# In[14]:



best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print("Test accuracy:", test_acc)

