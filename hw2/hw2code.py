import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin


def find_best_split(feature_vector, target_vector):
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]
    
    unique_values = np.unique(sorted_features)
    
    if len(unique_values) == 1:
        return np.array([]), np.array([]), None, None
    
    thresholds = (unique_values[:-1] + unique_values[1:]) / 2
    
    n = len(sorted_targets)
    
    split_indices = np.searchsorted(sorted_features, thresholds, side='right')
    
    class1_cumsum = np.cumsum(sorted_targets == 1)
    
    n_left = split_indices
    n_right = n - n_left
    
    n_left_class1 = class1_cumsum[split_indices - 1]
    n_right_class1 = np.sum(sorted_targets == 1) - n_left_class1
    
    p_left_1 = n_left_class1 / n_left
    p_left_0 = 1 - p_left_1
    p_right_1 = n_right_class1 / n_right
    p_right_0 = 1 - p_right_1
    
    H_left = 1 - p_left_1**2 - p_left_0**2
    H_right = 1 - p_right_1**2 - p_right_0**2
    
    ginis = -(n_left / n) * H_left - (n_right / n) * H_right
    
    valid_mask = (n_left > 0) & (n_right > 0)
    ginis[~valid_mask] = -np.inf
    
    best_idx = np.argmax(ginis)
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]
    
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, feature_types=None, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        self.feature_types = feature_types
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        if self.feature_types is not None:
            if np.any(list(map(lambda x: x != "real" and x != "categorical", self.feature_types))):
                raise ValueError("There is unknown feature type")
            self._feature_types = self.feature_types
        else:
            self._feature_types = ["real"] * X.shape[1]
        
        self._tree = {}
        self._fit_node(X, y, self._tree, depth=0)
        return self

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if self.max_depth is not None and depth >= self.max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        if self.min_samples_split is not None and len(sub_y) < self.min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {key: (clicks.get(key, 0) / count) for key, count in counts.items()}
                sorted_categories = [x[0] for x in sorted(ratio.items(), key=lambda x: x[1])]
                categories_map = {cat: idx for idx, cat in enumerate(sorted_categories)}
                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError

            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [cat for cat, idx in categories_map.items() if idx < threshold]
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        if self.min_samples_leaf is not None:
            n_left = np.sum(split)
            n_right = len(split) - n_left
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                node["type"] = "terminal"
                node["class"] = Counter(sub_y).most_common(1)[0][0]
                return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature_idx = node["feature_split"]
        feature_type = self._feature_types[feature_idx]
        
        if feature_type == "real":
            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])
