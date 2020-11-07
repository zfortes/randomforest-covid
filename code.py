import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

df = pd.read_excel('dataset.xlsx')

df['SARS-Cov-2 exam result'] = [0 if a == 'negative' else 1 for a in df['SARS-Cov-2 exam result'].values]

Y = df['SARS-Cov-2 exam result']

df = df.drop([
    "SARS-Cov-2 exam result",
    "Patient ID",
    'Patient addmited to regular ward (1=yes, 0=no)',
    'Patient addmited to semi-intensive unit (1=yes, 0=no)',
    'Patient addmited to intensive care unit (1=yes, 0=no)'
], axis=1)

df = df.fillna(df.mean())

# Fill NaNs with -10
df = df.fillna(-10, axis=1)

categorical_features = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object'] ]

X = pd.get_dummies(df, prefix=categorical_features, columns=categorical_features)

clf = RandomForestClassifier(max_depth=50, random_state=0, n_estimators=40)
# K = 10
# folds = KFold(K, shuffle=True, random_state=40)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


    # X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    # y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.40, random_state=55)


clf.fit(X_train, y_train)

pred_y = clf.predict(X_test)

print(classification_report(y_test, pred_y))
pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(f'Mean accuracy score: {accuracy:.3}')

#
# for fold, (train_index, test_index) in enumerate(folds.split(X, Y)):
#     print('Fold:', fold + 1)
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
#
#     clf.fit(X_train, y_train)
#
#     pred_y = clf.predict(X_test)
#
#     print(classification_report(y_test, pred_y))
#     pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, pred)
#     print(f'Mean accuracy score: {accuracy:.3}')
#
