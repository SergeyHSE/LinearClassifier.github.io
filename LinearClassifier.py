# -*-         coding: utf-8 -*-
"""
Original file is located at 'https://colab.research.google.com/drive/11IltBc0sFWhQ5dLnv5QMFwDsZ8mpOum_'
"""

import numpy as np
import pandas as pd

data = pd.read_csv('https://docs.google.com/uc?export=download&id=1VPkoWfiIvZl4HGp49BUaVEEblVIGYh91')
data.head()

"""
What percentage of the patients presented in the data have heart disease (`target' = 1`)?
"""

data['target'].value_counts()
d = 165
f = 138
h = (d / (d + f)) * 100
print(h)

# Let's split the data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'],
                                                    test_size=0.25, random_state=13)
X_train.shape, X_test.shape



"""
Let's train a linear classifier from `sklearn` with the maximum number of iterations `max_iter=1000`,
a constant value of the gradient descent step (`learning_rate='constant`) equal to 'eta0=0.1'.
As a seed, we will put `random_state=13'. Disable the regularization parameter: `alpha=0`.
The 'sklearn.linear_model.SGDClassifier' class combines different linear models - to get a logistic regression,
we fix the parameter `loss='log".

We need to find the value of the free coefficient of the resulting linear classifier
"""

from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss='log_loss',  alpha=0,  max_iter=1000, learning_rate='constant',
                    eta0=0.1, random_state=13)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

y_pred.shape

clf.coef_, clf.intercept_
d = list(clf.coef_)
print(d)

np.sqrt(np.sum(np.square(clf.coef_)))

"""
Now let's find the value of the L2-norm of the vector of weights
(without taking into account the free coefficient) of the resulting linear classifier

"""

def L2_norma(x):
  return np.sqrt(np.sum(np.square(x)))
L2 = L2_norma(clf.coef_)
print(L2)

"""
Let's find accuracy score
"""

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

"""
Let's train the classifier anew with the regularization parameter `alpha=0.1` and the regularization type `penalty='l1'.
Also, instead of a constant value of the gradient descent step, we will use the optimal one (`learning_rate='optimal').
In this case, the value of the initial step of gradient descent `eta0` does not participate in training in any way.
We do this to find out if the quality of the resulting classifier differs from the quality of the first one.
"""

clf_lin = SGDClassifier(loss='log', penalty='l1', alpha=0.1, max_iter=1000, random_state=13, learning_rate='optimal')
clf_lin.fit(X_train, y_train)
y_pred_l = clf_lin.predict(X_test)
accuracy_score(y_test, y_pred_l)

"""
Let's find the L2 norm of the weight vector for the resulting classifier.
We will see that the vector has become more sparse, and zero elements have appeared in it.
"""

L2_l = L2_norma(clf_lin.coef_)
print(L2_l)

""" 
Let's check how the resulting classifiers predict not classes, but the probabilities of classes,
and calculate the AUC-ROC value
"""

from sklearn.linear_model import LogisticRegression

y_pred_proba = clf.predict_proba(X_test)
y_pred_proba

y_pred_proba_d = np.delete(y_pred_proba, 0, 1)
y_pred_proba_d

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_proba_d)

"""
Calculate AUC_ROC value for two classificator
"""

y_pred_proba_l1 = clf_lin.predict_proba(X_test)
y_pred_proba_l1

y_pred_proba_d2 = np.delete(y_pred_proba_l1, 0, 1)
y_pred_proba_d2

roc_auc_score(y_test, y_pred_proba_d2)

"""
Finally, let's find out which feature is the most important in the opinion of the best model (has the largest coefficient modulo).
"""

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred_l)

T = pd.DataFrame(clf_lin.coef_)
T
D = pd.DataFrame(X_train.columns)
T = T.T
T

np.concatenate((D, T), axis=1)
