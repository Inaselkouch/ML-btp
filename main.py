from joblib import Parallel, delayed
import numpy as np
from graphviz import Digraph
from performance import zero_one_loss

#PART I

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        feature: the index of the feature to split on.
        thr: the threshold to split the feature on.
        left: the left child node.
        right: the right child node.
        value: the value of the leaf node.

        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # value of the node only if it is a leaf

        

    def is_leaf(self) -> bool:
        return self.value is not None
    
    """
    Returns True if the node has a value --> it's a leaf.
    Returns False if the node doesn't have a value --> it's an internal node.

    """

class DecisionTree:
    def __init__(self, max_depth=None, max_leaf_nodes=None, entropy_threshold=None, split_function='gini', min_samples_split=2, feature_names=None):
        """
        max_depth: the maximum depth of the tree (controls overfitting).
        max_leaf_nodes: the maximum number of leaf nodes (controls tree size).
        entropy_threshold: the threshold for entropy to stop splitting.
        split_function: the criterion to use for splitting.
        min_samples_split: the minimum number of samples required to split (by default it's 2).
        root: the root node of the tree.
        feature_names: the names of the features.
        leaf_count: the number of leaf nodes.
        depth: the depth of the tree.
        criterion_func: the criterion function to use for splitting.
        
        """
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.entropy_threshold = entropy_threshold
        self.split_function = split_function
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names = feature_names
        self.leaf_count = 0
        self.depth = 0
        self.criterion_func = {
            'scaled_entropy': self._scaled_entropy,
            'gini': self._gini_impurity,
            'squared': self._squared_impurity,
        }.get(self.split_function)
        self.feature_types = None

    def _get_feature_type(self, X):
         return ['numerical' if np.issubdtype(X[col].dtype, np.number) else 'categorical' for col in X.columns]

    def get_params(self):
        return {
            'max_depth': self.max_depth,
            'max_leaf_nodes': self.max_leaf_nodes,
            'entropy_threshold': self.entropy_threshold,
            'split_function': self.split_function,
            'min_samples_split': self.min_samples_split,
            'feature_names': self.feature_names
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
    
    def fit(self, X, y):
        self.feature_types = self._get_feature_type(X)
        self.root = self.grow_tree(X, y)
        return self
    
    def grow_tree(self, X, y, depth=0):
        if (self.max_depth is not None and depth >= self.max_depth) or (len(np.unique(y)) == 1) or (len(X) < self.min_samples_split) or (self.max_leaf_nodes is not None and self.leaf_count >= self.max_leaf_nodes):
            return Node(value=self._most_common_label(y))

        best_feature, best_threshold = self._best_criteria(X, y)
        if best_feature is None:
            return Node(value=self._most_common_label(y))
        if self.feature_types[best_feature] == 'categorical':
            left_indices, right_indices = self._split_cat(X.iloc[:, best_feature], best_threshold)
        else:
            left_indices, right_indices = self._split_num(X.iloc[:, best_feature], best_threshold)
        left = self.grow_tree(X.loc[left_indices], y.loc[left_indices], depth + 1)
        right = self.grow_tree(X.loc[right_indices], y.loc[right_indices], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _best_criteria(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        def evaluate_feature(feature):
            local_best_gain = -1
            local_best_threshold = None
            if self.feature_types[feature] == 'categorical':    
                thresholds = np.unique(X.iloc[:, feature])
            else:
                thresholds = np.percentile(X.iloc[:, feature], np.arange(0, 100, 10))

            for threshold in thresholds:
                if self.feature_types[feature] == 'categorical':
                    left_indices, right_indices = self._split_cat(X.iloc[:, feature], threshold)
                else:
                    left_indices, right_indices = self._split_num(X.iloc[:, feature], threshold)

                if len(left_indices) < self.min_samples_split or len(right_indices) < self.min_samples_split:
                    continue

                splits = [left_indices, right_indices]
                gain = self._information_gain(y, splits)

                if gain > local_best_gain:
                    local_best_gain = gain
                    local_best_threshold = threshold

            return feature, local_best_gain, local_best_threshold

        results = Parallel(n_jobs=-1)(delayed(evaluate_feature)(feature) for feature in range(X.shape[1]))

        # Find best result between al features 
        for feature, gain, threshold in results:
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

        return best_feature, best_threshold

    def _split_num(self, X, threshold):
        left_indices = X.index[X <= threshold]
        right_indices = X.index[X > threshold]
        return left_indices, right_indices


    def _split_cat(self, X, threshold):
        left_indices = X.index[X == threshold]
        right_indices = X.index[X != threshold]
        return left_indices, right_indices
    
    def _information_gain(self, y, splits):
        total_entropy = self.criterion_func(y)
        splits_entropy = np.sum([len(split) / len(y) * self.criterion_func(y.loc[split]) for split in splits])
        return total_entropy - splits_entropy
    
    def _gini_impurity(self, y):
        prob = y.value_counts(normalize=True).iloc[0]
        return 2*prob*(1-prob)
    
    def _scaled_entropy(self, y):
        prob = y.value_counts(normalize=True).iloc[0]
        if prob == 0 or prob == 1:
            return 0
        return -prob/2 * np.log2(prob) - (1-prob)/2 * np.log2(1-prob) 
    
    def _squared_impurity(self, y):
        prob = y.value_counts(normalize=True).iloc[0]
        return np.sqrt(prob * (1 - prob))

    def _most_common_label(self, y):
        return y.value_counts().index[0]
    
    def predict(self, X):
        return [self._predict_single(x, self.root) for _, x in X.iterrows()]
    
    def _predict_single(self, x, node):
        if node.is_leaf():
            return node.value
        if self.feature_types[node.feature] == 'categorical':
            if x.iloc[node.feature] == node.threshold:
                return self._predict_single(x, node.left)
            return self._predict_single(x, node.right)
        else:
            if x.iloc[node.feature] <= node.threshold:
                return self._predict_single(x, node.left)
            return self._predict_single(x, node.right)








