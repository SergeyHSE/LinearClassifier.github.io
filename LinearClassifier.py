# -*-         coding: utf-8 -*-
"""
Data contains;

age - age in years
sex - (1 = male; 0 = female)
cp - chest pain type
trestbps - resting blood pressure (in mm Hg on admission to the hospital)
chol - serum cholestoral in mg/dl
fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg - resting electrocardiographic results
thalach - maximum heart rate achieved
exang - exercise induced angina (1 = yes; 0 = no)
oldpeak - ST depression induced by exercise relative to rest
slope - the slope of the peak exercise ST segment
ca - number of major vessels (0-3) colored by flourosopy
thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
target - have disease or not (1=yes, 0=no)
"""

import numpy as np
import pandas as pd

data = pd.read_csv('https://docs.google.com/uc?export=download&id=1VPkoWfiIvZl4HGp49BUaVEEblVIGYh91')
data.head()

# build correlation matrix

plt.figure(figsize=(12, 10), dpi=100)
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()
   
# Build regplots

data.columns
feature_names = list(data.drop('target', axis=1).columns)
feature_names
len(feature_names)

num_rows = 7
num_cols = 2
num_plots = len(feature_names)
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 20))
axes = axes.flatten()
for i in range(num_plots):
    if i < num_plots:
        sns.regplot(x=feature_names[i], y='target', data=data, ax=axes[i])
        axes[i].set_title(f'Regression Plot for {feature_names[i]}')
    else:
        fig.delaxes(axes[i])
plt.tight_layout()
plt.show()

# Build boxplots

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 20))
axes = axes.flatten()
for i in range(num_plots):
    if i < num_plots:
        sns.boxplot(x=feature_names[i], y='target', data=data, ax=axes[i])
        axes[i].set_title(f'Regression Plot for {feature_names[i]}')
    else:
        fig.delaxes(axes[i])
plt.tight_layout()
plt.show()

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
