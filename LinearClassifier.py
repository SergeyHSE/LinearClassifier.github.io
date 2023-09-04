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

clf = SGDClassifier(loss='log',  alpha=0,  max_iter=1000, learning_rate='constant',
                    eta0=0.1, random_state=13)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

y_pred.shape

clf.coef_, clf.intercept_
d = list(clf.coef_)
print(d)

np.sqrt(np.sum(np.square(clf.coef_)))

"""3. Какое значение L2-нормы вектора весов (без учета свободного коэффициента) у полученного линейного классификатора? Ответ округлите до двух знаков после запятой.

_Напоминание. L2-норма вектора $v = (v_1, \ldots, v_n)$ - это корень из суммы квадратов его элементов:_

$$
\|v\|_2 = \sqrt{\sum\limits_{i=1}^nv_i^2}
$$
"""

def L2_norma(x):
  return np.sqrt(np.sum(np.square(x)))
L2 = L2_norma(clf.coef_)
print(L2)

"""4. Найдите долю правильных ответов классификатора на тестовой части выборки **(в процентах)**. Ответ округлите до двух знаков после запятой. Например, если значение доли правильных ответов будет равно 0.1234, то ответом будет 12.34 - ведь это 12.34%."""

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

"""5. В задаче классификации, как и в задаче регрессии, для оптимизации линейных моделей можно применять регуляризацию. Этот метод реализован и в `sklearn.linear_model.SGDClassifier` - параметр регуляризации обозначается параметром `alpha`. За тип регуляризации (L1, L2 или обе сразу) отвечает параметр `penalty`.

   Обучите классификатор заново с параметром регуляризации `alpha=0.1` и типом регуляризации `penalty='l1'`. Оставьте максимальное число итераций, равное `max_iter=1000` и сид `random_state=13`. Также вместо постоянного значения шага градиентного спуска используйте оптимальное (`learning_rate='optimal'`), которое, кстати, зависит от `alpha` (о том, как именно он вычисляется и какие еще параметры можно выбрать, можно подробнее прочитать в [документации](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)). В данном случае значение начального шага градиентного спуска `eta0` никак не участвует в обучении.
   
   Отличается ли качество полученного классификатора от качества первого? Какая доля правильных ответов получается теперь на тестовой выборке? Выразите ее **в процентах**, ответ округлите до двух знаков после запятой. Например, если значение доли правильных ответов будет равно 0.1234, то ответом будет 12.34 - ведь это 12.34%.
"""

clf_lin = SGDClassifier(loss='log', penalty='l1', alpha=0.1, max_iter=1000, random_state=13, learning_rate='optimal')
clf_lin.fit(X_train, y_train)
y_pred_l = clf_lin.predict(X_test)
accuracy_score(y_test, y_pred_l)

"""6. Найдите L2-норму вектора весов для полученного классификатора (заметьте, как на нее повлияла регуляризация). Ответ округлите до двух знаков после запятой.

    Заметьте, что вектор стал более разреженным, и в нем появились нулевые элементы - это результат действия L1-регуляризации.
"""

L2_l = L2_norma(clf_lin.coef_)
print(L2_l)

"""7. Наконец, проверьте, как полученные классификаторы предсказывают не классы, а вероятности классов - так как мы работаем с логистической регрессией, это можно сделать. Посмотрите на вероятности, которые выдает первый классификатор (обученный с постоянным шагом градиентного спуска и без регуляризации) на тестовой части выборки. В этом вам поможет метод `predict_proba`. Результатом его работы будет список размера $N\times 2$, где $N$ - это число объектов. В каждом столбце списка находятся вероятности соответствующего класса для объектов. Поэтому если вам нужен положительный класс, вас интересует последний столбец.

    Какое получается значение AUC-ROC? Ответ округлите до двух знаков после запятой.
"""

from sklearn.linear_model import LogisticRegression

y_pred_proba = clf.predict_proba(X_test)
y_pred_proba

y_pred_proba_d = np.delete(y_pred_proba, 0, 1)
y_pred_proba_d

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_proba_d)

"""8. Посмотрите на вероятности, которые выдает второй классификатор (обученный с оптимальным шагом градиентного спуска и с регуляризацией) на тестовой части выборки. Что вы наблюдаете - как отличаются эти вероятности от вероятностей первого классификатора?

   Посчитайте значение AUC-ROC второго классификатора. Ответ округлите до двух знаков после запятой.
"""

y_pred_proba_l1 = clf_lin.predict_proba(X_test)
y_pred_proba_l1

y_pred_proba_d2 = np.delete(y_pred_proba_l1, 0, 1)
y_pred_proba_d2

roc_auc_score(y_test, y_pred_proba_d2)

"""9. Какой признак является самым важным по мнению лучшей модели (имеет наибольший по модулю коэффициент) для принятия решения?"""

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred_l)

T = pd.DataFrame(clf_lin.coef_)
T
D = pd.DataFrame(X_train.columns)
T = T.T
T

np.concatenate((D, T), axis=1)