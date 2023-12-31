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

# Build bar plots

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 18))
axes = axes.flatten()
for i, feature in enumerate(feature_names):
    if i < len(feature_names):
        sns.countplot(x=feature, hue='target', data=data, ax=axes[i])
        axes[i].set_title(f'Count Plot for {feature}')
plt.tight_layout()
plt.show()

# Build violinplots

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(16, 16))
axes = axes.flatten()
for i, feature in enumerate(feature_names):
    if i < len(feature_names):
        sns.violinplot(x='target', y=feature, data=data, ax=axes[i])
        axes[i].set_title(f'Violin Plot for {feature}')
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

# Build countplot in percent
    
plt.figure(figsize=(7, 6), dpi=100)
total = float(len(data))
ax = sns.countplot(x='target', data=data)
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{(height/total)*100:.2f}%', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=14, color='black')
plt.show()

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

accuracy = accuracy_score(y_test, y_pred)
accuracy

# Build accuracy score

plt.figure(figsize=(6, 6), dpi=100)
plt.bar(['Accuracy'], [accuracy], color='orange')
plt.ylabel('Accuracy')
plt.title('Accuracy Score')
plt.ylim(0, 1)
plt.text(0, accuracy, f'{accuracy:.2f}', ha='center', va='bottom', fontsize=16, color='black')
plt.show()

"""
Let's train the classifier anew with the regularization parameter `alpha=0.1` and the regularization type `penalty='l1'.
Also, instead of a constant value of the gradient descent step, we will use the optimal one (`learning_rate='optimal').
In this case, the value of the initial step of gradient descent `eta0` does not participate in training in any way.
We do this to find out if the quality of the resulting classifier differs from the quality of the first one.
"""

clf_lin = SGDClassifier(loss='log', penalty='l1', alpha=0.1, max_iter=1000, random_state=13, learning_rate='optimal')
clf_lin.fit(X_train, y_train)
y_pred_l = clf_lin.predict(X_test)
accuracy_l = accuracy_score(y_test, y_pred_l)

plt.figure(figsize=(6, 6), dpi=100)
plt.bar(['Accuracy'], [accuracy_l], color='brown')
plt.ylabel('Accuracy')
plt.title('Accuracy Score with regularizator')
plt.ylim(0, 1)
plt.text(0, accuracy_l, f'{accuracy:.2f}', ha='center', va='bottom', fontsize=16, color='black') 
plt.show()

# Compare two accuracy

plt.figure(figsize=(12, 6), dpi=100)
plt.subplot(1, 2, 1)
plt.bar(['Accuracy'], [accuracy], color='orange')
plt.ylabel('Accuracy')
plt.title('Accuracy Score')
plt.ylim(0, 1)
plt.text(0, accuracy, f'{accuracy:.2f}', ha='center', va='bottom', fontsize=16, color='black')
plt.subplot(1, 2, 2)
plt.bar(['Accuracy'], [accuracy_l], color='brown')
plt.ylabel('Accuracy')
plt.title('Accuracy Score with regularizator')
plt.ylim(0, 1)
plt.text(0, accuracy_l, f'{accuracy_l:.2f}', ha='center', va='bottom', fontsize=16, color='black')
plt.tight_layout()
plt.show()

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

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_d)

roc_auc = roc_auc_score(y_test, y_pred_proba_d)

# Plot ROC curve
plt.figure(figsize=(8, 6), dpi=100)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', color='red', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

"""
Calculate AUC_ROC value for two classificator
"""

y_pred_proba_l1 = clf_lin.predict_proba(X_test)
y_pred_proba_l1

y_pred_proba_d2 = np.delete(y_pred_proba_l1, 0, 1)
y_pred_proba_d2

roc_auc2 = roc_auc_score(y_test, y_pred_proba_d2)
fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred_proba_d2)

# Plot ROC curve for second classifier
plt.figure(figsize=(8, 6), dpi=100)
plt.plot(fpr2, tpr2, label=f'ROC Curve (AUC = {roc_auc2:.2f})')
plt.plot([0, 1], [0, 1], 'k--', color='red', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve for second Classifier')
plt.legend(loc="lower right")
plt.show()

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

# Build figure

coefficients_df = pd.concat([D, T], axis=1)
coefficients_df.columns = ['Feature', 'Coefficient']
coefficients_df

# Sort the DataFrame by coefficient magnitude (absolute value) for better visualization
coefficients_df['Coefficient_ABS'] = np.abs(coefficients_df['Coefficient'])
coefficients_df = coefficients_df.sort_values(by='Coefficient_ABS', ascending=False)

# Create a bar plot to visualize coefficients
plt.figure(figsize=(10, 6), dpi=100)
plt.barh(coefficients_df['Feature'], coefficients_df['Coefficient'], color='blue')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Logistic Regression Coefficients')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()
