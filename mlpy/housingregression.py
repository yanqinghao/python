import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

df = pd.read_csv('housing.data', header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print df.head()
sns.set(style='whitegrid', context='notebook')
# sns.reset_orig()
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.show()

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
plt.show()

class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros((1 + X.shape[1],1))
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output.reshape(output.size,1))
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None
if __name__=='__main__':
    X = df[['RM']].values
    y = df[['MEDV']].values
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    X_std = sc_x.fit_transform(X)
    y_std = sc_y.fit_transform(y)
    lr = LinearRegressionGD()
    lr.fit(X_std, y_std)
    plt.plot(range(1, lr.n_iter+1), lr.cost_)
    plt.ylabel('SSE')
    plt.xlabel('Epoch')
    plt.show()

    lin_regplot(X_std, y_std, lr)
    plt.xlabel('Average number of rooms [RM] (standardized)')
    plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
    plt.show()

    num_rooms_std = sc_x.transform([[5.0]])
    price_std = lr.predict(num_rooms_std)
    print("Price in $1000's: %.3f" % sc_y.inverse_transform(price_std))
    print('Slope: %.3f' % lr.w_[1])
    print('Intercept: %.3f' % lr.w_[0])

    slr = LinearRegression()
    slr.fit(X, y)
    print('Slope: %.3f' % slr.coef_[0])
    print('Intercept: %.3f' % slr.intercept_)

    lin_regplot(X, y, slr)
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000\'s [MEDV]')
    plt.show()

    Xb = np.hstack((np.ones((X.shape[0], 1)), X))
    w = np.zeros(X.shape[1])
    z = np.linalg.inv(np.dot(Xb.T, Xb))
    w = np.dot(z, np.dot(Xb.T, y))
    print('Slope: %.3f' % w[1])
    print('Intercept: %.3f' % w[0])

    ransac = RANSACRegressor(LinearRegression(), max_trials=100, min_samples=50,
                             loss=lambda x1,x2: np.sum(np.abs(x1-x2), axis=1), residual_threshold=5.0, random_state=0)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    line_X = np.arange(3, 10, 1)
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])
    plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
    plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label='Outliers')
    plt.plot(line_X, line_y_ransac, color='red')
    plt.xlabel('Average number of rooms [RM]')
    plt.ylabel('Price in $1000\'s [MEDV]')
    plt.legend(loc='upper left')
    plt.show()
    print('Slope: %.3f' % ransac.estimator_.coef_[0])
    print('Intercept: %.3f' % ransac.estimator_.intercept_)

    X = df.iloc[:, :-1].values
    y = df['MEDV'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    slr = LinearRegression()
    slr.fit(X_train, y_train)
    y_train_pred = slr.predict(X_train)
    y_test_pred = slr.predict(X_test)
    plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
    plt.scatter(y_test_pred, y_test_pred - y_test, c='lightgreen', marker='s', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
    plt.xlim([-10, 50])
    plt.show()
    print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=1.0)
    elastic = ElasticNet(alpha=1.0, l1_ratio=0.5)