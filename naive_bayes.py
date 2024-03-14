#%%
import numpy as np
from sklearn.model_selection import train_test_split

class NaiveBayes():
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes,  dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[c==y]
            self._mean[idx,:] = X_c.mean(axis=0)
            self._var[idx,:] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)


    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            with np.errstate(divide='ignore'):
                class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        
        return self._classes[np.argmax(posteriors)]


    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x-mean)**2 / (2*var))
        denominator = np.sqrt(2*np.pi*var)
        return numerator/denominator
    

#%%
if __name__ == '__main__':
    data = np.loadtxt('data/winequality-red.csv', delimiter=',', skiprows=1)
    X, y = data[:,:-1], data[:,-1]

    X_train, X_test, y_train, y_test =  train_test_split(X, y, 
        test_size=0.2, 
        random_state=1234)

    NB = NaiveBayes()
    NB.fit(X_train, y_train)
    predictions = NB.predict(X_test)

    acc = np.sum(predictions == y_test) / len(y_test)
    print(f'Accuracy: {acc*100}%')