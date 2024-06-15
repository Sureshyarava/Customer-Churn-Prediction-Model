import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


#load data
df = pd.read_csv("customer_churn_prediction.csv")

print(df.head())

print(df.info())

# check null values
print(df.isna().sum())

# class imbance check
unique_values_counts_churn = df['churn'].value_counts()

print(unique_values_counts_churn)

df.duplicated().sum()

# one hot encoding
categorical_columns = ['country' , 'gender']

encoded_df = pd.get_dummies(df, columns=categorical_columns, dtype=int)

# drop irrelevant column
data = encoded_df.drop('customer_id', axis=1)

print(data.head())

# split feature data and target data
y = data['churn']
X = data.drop('churn', axis=1)

# logic for sampling and splitting the data
def sampling(X, y, test_size=0.5):
      data = pd.concat([X, y], axis=1)
      data_1 = data[data['churn'] == 1]
      data_0 = data[data['churn'] == 0]

      y_1 = data_1['churn']
      X_1 = data_1.drop('churn', axis=1)

      y_0 = data_0['churn']
      X_0 = data_0.drop('churn', axis=1)

      X_1_sample = X_1.sample(n=1000, random_state=42)
      y_1_sample = y_1.loc[X_1_sample.index]

      X_0_sample = X_0.sample(n=1000, random_state=42)
      y_0_sample = y_0.loc[X_0_sample.index]

      X_sample = pd.concat([X_1_sample, X_0_sample])

      y_sample = pd.concat([y_1_sample, y_0_sample])

      # stratified sampling
      X_val, X_test, y_val, y_test = train_test_split(X_sample, y_sample, test_size = test_size, stratify=y_sample, random_state=42)


      indices_X_1_remaining = X_1.index.difference(X_1_sample.index)

      indices_X_0_remaining = X_0.index.difference(X_0_sample.index)

      X_1_remaining = X_1.loc[indices_X_1_remaining]
      y_1_remaining = y_1.loc[indices_X_1_remaining]

      X_0_remaining = X_0.loc[indices_X_0_remaining]
      y_0_remaining = y_0.loc[indices_X_0_remaining]

      X_train = pd.concat([X_1_remaining, X_0_remaining])
      y_train = pd.concat([y_1_remaining, y_0_remaining])

      print(f'X_val shape is {X_val.shape}')
      print(f'X_test shape is {X_test.shape}')
      print(f'X_train shape is {X_train.shape}')

      return X_val, X_test, X_train, y_val, y_test, y_train

# train, test, validation split
X_val, X_test, X_train, y_val, y_test, y_train = sampling(X,y)


# Standardized Scaler
scaler = StandardScaler()
X_train[X_train.columns] = scaler.fit_transform(X_train[X_train.columns])

X_val[X_val.columns] = scaler.fit_transform(X_val[X_val.columns])

X_test[X_test.columns] = scaler.fit_transform(X_test[X_test.columns])

# class imbalance resampling
smote = SMOTE()
X_train,y_train = smote.fit_resample(X_train,y_train)
print(Counter(y_train))


def compute_f1_score(model, x_data, labels):
  pred_y = model.predict(x_data)
  return f1_score(labels, pred_y)


def compute_roc_auc_score(model, x_data, labels):
  pred_y = model.predict_proba(x_data)[:, 1]
  return roc_auc_score(labels, pred_y)

# Logistic regression
lr = LogisticRegression()

lr.fit(X_train, y_train)

# f1 scores
print(compute_f1_score(lr, X_val, y_val))
print(compute_f1_score(lr, X_test, y_test))

# roc auc scores
print(compute_roc_auc_score(lr, X_val, y_val))
print(compute_roc_auc_score(lr, X_test, y_test))


# Decision trees
clf = DecisionTreeClassifier(random_state=42, max_depth=15, max_features = 5)

clf.fit(X_train, y_train)

# metrics
print(compute_f1_score(clf, X_val, y_val))
print(compute_f1_score(clf, X_test, y_test))
print(compute_roc_auc_score(clf, X_val, y_val))
print(compute_roc_auc_score(clf, X_test, y_test))


#random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth = 9, min_samples_leaf = 2)

rf.fit(X_train, y_train)

compute_f1_score(rf, X_val, y_val)
compute_roc_auc_score(rf, X_val, y_val)

# hyper parameter tuning for Random forest
def search_hyperparams_and_train_for_RF(train_x, train_y, val_x, val_y):
    best_hyperparams = {}
    best_model = None

    max_depths = [9, 10, 11, 12]
    min_samples_leaf = [1, 2, 4]
    n_estimators = [100, 150, 200, 250]

    best_score = 0
    for depth in max_depths:
        for min_leaf in min_samples_leaf:
            for n_estimator in n_estimators:
              model = RandomForestClassifier(n_estimators=n_estimator, random_state=42, max_depth = depth, min_samples_leaf = min_leaf)
              hypp_str = 'max_depth=\'{}\', min_samples_leaf={}, n-estimators={}'.format(depth, min_leaf, n_estimator)

              model = model.fit(train_x, train_y)
              score = compute_roc_auc_score(model, val_x, val_y)

              if score > best_score:
                  best_hyperparams = hypp_str
                  best_score = score
                  best_model = model

    return best_model, best_hyperparams

best_rf_model, hyper_param_rf = search_hyperparams_and_train_for_RF(X_train, y_train, X_val, y_val)

print(hyper_param_rf)

# results
compute_roc_auc_score(best_rf_model, X_test, y_test)
compute_f1_score(best_rf_model, X_test, y_test)

print(compute_f1_score(best_rf_model, X_val, y_val))
print(compute_roc_auc_score(best_rf_model, X_val, y_val))

#XGBoost Classifier
xgb = XGBClassifier(objective='binary:logistic', n_estimators = 150, random_state=42, max_depth = 5, eta = 0.2)
xgb.fit(X_train, y_train)

compute_f1_score(xgb, X_val, y_val)
compute_roc_auc_score(xgb, X_val, y_val)

# hyper parameter tuning for XG boost
def search_hyperparams_and_train_for_XGB(train_x, train_y, val_x, val_y):
    best_hyperparams = {}
    best_model = None

    max_depths = [4, 5, 6, 7]
    etas = [0.1, 0.15, 0.2]
    n_estimators = [100, 150, 200]

    best_score = 0
    for depth in max_depths:
        for eta in etas:
            for n_estimator in n_estimators:
              model = XGBClassifier(objective='binary:logistic', n_estimators = n_estimator, random_state=42, max_depth = depth, eta = eta)
              hypp_str = 'max_depth=\'{}\', eta={}, n-estimators={}'.format(depth, eta, n_estimator)

              model = model.fit(train_x, train_y)
              score = compute_roc_auc_score(model, val_x, val_y)

              if score > best_score:
                  best_hyperparams = hypp_str
                  best_score = score
                  best_model = model

    return best_model, best_hyperparams

best_model_xgb, hyper_param_xgb = search_hyperparams_and_train_for_XGB(X_train, y_train, X_val, y_val)

print(hyper_param_xgb)

#results
print(compute_roc_auc_score(best_model_xgb, X_test, y_test))
print(compute_roc_auc_score(best_model_xgb, X_val, y_val))

print(compute_f1_score(best_model_xgb, X_test, y_test))
print(compute_f1_score(best_model_xgb, X_val, y_val))

# K nearest neighbours
knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train, y_train)

# results
compute_f1_score(knn, X_val, y_val)
compute_f1_score(knn, X_test, y_test)
compute_roc_auc_score(knn, X_val, y_val)
compute_roc_auc_score(knn, X_test, y_test)

