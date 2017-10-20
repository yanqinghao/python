# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:15:37 2017

@author: yqh
"""
import AdalineSGD as adgd
import plot_decision as pltdcn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('E:/PyWorkSpace/spyder/iris/iris.txt',header = None)
df.tail
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

ada = adgd.AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada.fit(X_std, y)
pltdcn.plot_decision_regions(X_std, y, classifier=ada)
plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.show()
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()