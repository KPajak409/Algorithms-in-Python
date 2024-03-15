#%%
import numpy as np
from decision_tree import DecisionTree
from sklearn.model_selection import train_test_split
from collections import Counter

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    subset = X[idxs], y[idxs]
    return subset

def most_common_label( y):
    counter = Counter(y)
    # most common (n_most_common_el)[element_id_in_the_list][tuple 0: value 1: occurencies]
    most_common = counter.most_common(1)[0][0]
    return most_common

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

class RandomForest:
    def __init__(self, n_trees, min_samples_split=2, max_depth=100, n_features=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []


    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split,
                                max_depth=self.max_depth,
                                n_features=self.n_features)
            X_sample, y_sample = bootstrap_sample(X,y)
            tree.fit(X_sample,y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # majority vote
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred  = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
    

if __name__ == '__main__':
    data = np.loadtxt('data/winequality-red.csv', delimiter=',', skiprows=1)
    X, y = data[:,:-1], data[:,-1]

    X_train, X_test, y_train, y_test =  train_test_split(X, y, 
        test_size=0.2, 
        random_state=1234)

    r_forest = RandomForest(n_trees=100, min_samples_split=4, max_depth=100)
    r_forest.fit(X_train, y_train)

    y_pred = r_forest.predict(X_test)
    acc = accuracy(y_test, y_pred)
    print(acc)   


            
# %%
