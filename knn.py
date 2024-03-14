#%%
import numpy as np
from sklearn.model_selection import train_test_split

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1-x2)**2)


class KNN:
    def __init__(self, k=3):
        self.k = k


    def fit(self, X, y):
        self.X = X
        self.y = y


    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    

    def _predict(self, x):
        # compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X]

        # get k nearest samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y[i] for i in k_indices]

        # majority vote
        return self._most_frequent(k_nearest_labels)


    def _most_frequent(self,list):
        counter = 0
        num = list[0]
        
        for i in list:
            curr_frequency = list.count(i)
            if(curr_frequency> counter):
                counter = curr_frequency
                num = i
    
        return num
#%%
if __name__ == '__main__':
    # load data
    data = np.loadtxt('data/winequality-red.csv', delimiter=',', skiprows=1)
    X, y = data[:,:-1], data[:,-1]
    
    X_train, X_test, y_train, y_test =  train_test_split(X, y, 
        test_size=0.2, 
        random_state=1234)
    
    # classifying
    clf = KNN(k=10)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    acc = np.sum(predictions == y_test) / len(y_test)
    print(f'Accuracy: {acc}%')
