import itertools
import numpy as np
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from main import DecisionTree
from performance import zero_one_loss


def split_train_test(X, y, test_size=0.2, random_state=123):
    np.random.seed(random_state) 
    num_samples = len(X)
    indices = np.random.permutation(num_samples)
    split_index = int(num_samples * test_size) 
    train_indices = indices[split_index:]  
    test_indices = indices[:split_index]
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

def model_evaluation(y_true, y_pred, y_prob):
    accuracy = np.mean(y_true == y_pred)
    loss = log_loss(y_true, y_prob)
    return accuracy, loss

def hyperparameter_tuning(X, y, param_grid, test_size=0.2, random_state=123):
    best_score = np.inf
    best_params = None
    param_combinations = list(itertools.product(*param_grid.values()))
    for params in param_combinations:
        param_dict = dict(zip(param_grid.keys(), params))  # Converti la tupla in un dizionario
        print(param_dict)
        model = DecisionTree(**param_dict)
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = zero_one_loss(y_test, y_pred)
        print("Score: ", score)
        print("-" * 50)
        if score < best_score:
            best_score = score
            best_params = param_dict
    return best_params, best_score

######################################################



