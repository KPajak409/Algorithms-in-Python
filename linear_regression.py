#%%
import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        #init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted-y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)

#%%
if __name__ == '__main__':

    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

    regressor = LinearRegression(lr=0.01, n_iters=500)
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X)
    mse_value = mse(y, predicted)
    print(mse_value)

    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8,6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, predicted, color='black')
    plt.show()

    print(regressor.weights, regressor.bias)
