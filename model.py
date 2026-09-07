"""
Random Forest from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impurity
from collections import Counter

def impurity(labels):
    """Return a non-negative impurity score for a 1D array of integer class labels."""
    # TODO: score how mixed the labels are; 0 for a pure set, larger for more mixed sets.
    unique_labels_dict = Counter(labels)
    sum_pk = 0.0
    for label in unique_labels_dict:
        number = unique_labels_dict[label]
        sum_pk += (number/len(labels))**2

    return 1 - sum_pk

# Step 2 - split_dataset
import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    # TODO: partition rows into left (feature <= threshold) and right (feature > threshold)
    feature_row = features[:,feature_index]
    left_features = features[feature_row<=threshold,:]
    left_labels = labels[feature_row<=threshold]
    right_features = features[feature_row>threshold, :]
    right_labels = labels[feature_row>threshold]

    return (left_features, left_labels, right_features, right_labels)

# Step 3 - split_score
def split_score(parent_labels, left_labels, right_labels):
    # TODO: return a score where higher means the children are purer than the parent.
    left_w = len(left_labels)/len(parent_labels)
    left_r = len(right_labels)/len(parent_labels)
    return impurity(parent_labels) - left_w*impurity(left_labels) - left_r*impurity(right_labels)

# Step 4 - best_split
import numpy as np

def best_split(features, labels, feature_indices):
    # TODO: search feature_indices for the (feature, threshold) that best improves purity.
    dct ={'feature_index':-1, 'threshold':-1, 'score':-1}
    for index in feature_indices:
        feature_column = features[:,index]
        trashholds = np.unique(feature_column)
        for trashhold in trashholds:
            (left_features, left_labels, right_features, right_labels) = split_dataset(features, labels, index, trashhold)
            score = split_score(labels, left_labels, right_labels)
            if (score > dct['score']):
                dct['score'] = score
                dct['feature_index'] = index
                dct['threshold'] = trashhold
    return dct

# Step 5 - should_stop
def should_stop(labels, depth, max_depth, min_samples_split):
    """Return True if this node should become a leaf instead of splitting further."""
    # TODO: decide whether to stop growing based on purity, depth, and size...
    if (depth>= max_depth or len(labels)<= min_samples_split or impurity(labels)==0):
        return True
    return False

# Step 6 - leaf_prediction
from collections import Counter
def leaf_prediction(labels):
    # TODO: choose a single class label to output for a leaf given the labels that reached it
    classes_dict = Counter(labels)
    cls = classes_dict.most_common(1)[0][0]
    return cls.item()

# Step 7 - build_tree
def build_tree(features, labels, max_depth=10, min_samples_split=2, feature_subset=None, depth=0):
    # TODO: recursively grow a decision tree, returning a nested dict of leaf/internal nodes.
    if (should_stop(labels, depth, max_depth, min_samples_split)):
        prediction = leaf_prediction(labels)
        return {'leaf': True, 'prediction': prediction}
    else:
        if feature_subset is not None:
            feature_indices = feature_subset
        else:
            feature_indices = np.arange(features.shape[1])
        dct = best_split(features, labels, feature_indices)
        (left_features, left_labels, right_features, right_labels) = split_dataset(features, labels, dct['feature_index'], dct['threshold'])
        left_node = build_tree(left_features, left_labels, max_depth, min_samples_split, feature_subset, depth=depth+1)
        right_node =build_tree(right_features, right_labels, max_depth, min_samples_split, feature_subset, depth=depth+1)
        return {'leaf': False, 'feature_index': dct['feature_index'], 'threshold': dct['threshold'], 'left': left_node, 'right': right_node}

# Step 8 - predict_example_tree
def predict_example_tree(tree, example):
    # TODO: walk the example down the fitted tree until you reach a leaf, then return its prediction.
    if (tree['leaf']):
        return tree['prediction']
    else:
        if (example[tree['feature_index']] > tree['threshold']):
            return predict_example_tree(tree['right'], example)
        else:
            return predict_example_tree(tree['left'], example)

# Step 9 - predict_tree
def predict_tree(tree, features):
    """Predict class labels for every row of `features` using a fitted decision tree.

    tree: dict returned by build_tree
    features: np.ndarray of shape (n, d)
    returns: np.ndarray of shape (n,) with integer class labels
    """
    # TODO: return predicted class for each row of features using the fitted tree.
    preds = np.array([predict_example_tree(tree, example) for example in features])
    return preds

# Step 10 - bootstrap_sample
def bootstrap_sample(features, labels, rng):
    # TODO: draw a bootstrap sample of rows (with replacement) using rng.
    indices = rng.choice(len(features), size = len(features), replace =True)
    return (features[indices], labels[indices])

# Step 11 - feature_subset
import numpy as np

def feature_subset(num_features, num_to_pick, rng):
    # TODO: return num_to_pick distinct random feature indices from range(num_features) using rng.
    indices = rng.choice(num_features, size =num_to_pick, replace=False)
    return indices

# Step 12 - train_forest
import numpy as np
from numpy.random import RandomState

def train_forest(features, labels, num_trees=10, max_depth=10, min_samples_split=2, num_features_per_split=None, random_state=0):
    # TODO: grow num_trees decision trees on bootstrap samples with random feature subsets.
    trees = []
    if num_features_per_split is None:
        num_features_per_split = int(np.round(np.sqrt(features.shape[1])))
    rng = np.random.RandomState(random_state)
    for i in range(num_trees):
        
        sampled_features, sampled_labels = bootstrap_sample(features, labels, rng)
        selected_indices = feature_subset(features.shape[1], num_features_per_split, rng)
        tree = build_tree(sampled_features, sampled_labels, max_depth, min_samples_split, selected_indices, 0)
        trees.append({'tree':tree, 'feature_indices':selected_indices})
    return trees

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy
def accuracy(predictions, labels):
    # TODO: compute the fraction of entries where predictions equals labels
    return (np.sum(predictions == labels))/len(labels)

