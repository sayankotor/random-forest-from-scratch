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

# Step 6 - leaf_prediction (not yet solved)
# TODO: implement

# Step 7 - build_tree (not yet solved)
# TODO: implement

# Step 8 - predict_example_tree (not yet solved)
# TODO: implement

# Step 9 - predict_tree (not yet solved)
# TODO: implement

# Step 10 - bootstrap_sample (not yet solved)
# TODO: implement

# Step 11 - feature_subset (not yet solved)
# TODO: implement

# Step 12 - train_forest (not yet solved)
# TODO: implement

# Step 13 - combine_predictions (not yet solved)
# TODO: implement

# Step 14 - predict_forest (not yet solved)
# TODO: implement

# Step 15 - accuracy (not yet solved)
# TODO: implement

