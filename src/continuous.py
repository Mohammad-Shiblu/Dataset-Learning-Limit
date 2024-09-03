import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier, export_text, _tree
from scipy import stats

def calculate_continuous_ambiguity(df, class_column):
    """
    Calculate ambiguity for continuous featured datasets using Recatangular hypercube encasing method.
    Args:
        df (pandas dataframe): Dataset
        class_column (string): class column name
    returns:
        float: ambiguity for conitnuous case
    """
    features = df.columns.drop(class_column)
    classes = df[class_column].unique()

    # Calculate Hypercubes
    grouped = df.groupby(class_column)[features]
    min_values = grouped.min()
    max_values = grouped.max()
    hypercubes = {cls: (min_values.loc[cls].values, max_values.loc[cls].values) for cls in classes}

    # Calculate Overlap Regions for all combinations of classes
    overlap_regions = []
    class_list = list(hypercubes.keys())
    num_classes = len(class_list)
    
    # Find all possible combinations of overlapping classes
    for r in range(2, num_classes + 1):
        for class_comb in combinations(class_list, r):
            min_overlap = np.maximum.reduce([hypercubes[cls][0] for cls in class_comb])
            max_overlap = np.minimum.reduce([hypercubes[cls][1] for cls in class_comb])
            if np.all(min_overlap <= max_overlap):
                overlap_regions.append((class_comb, min_overlap, max_overlap))

    # Count Samples in Overlap Regions
    samples = df[features].values
    class_labels = df[class_column].values

    samples_in_overlap = {}
    counted_samples = np.zeros(samples.shape[0], dtype=bool)  # To keep track of counted samples

    # Initialize overlap count for each class
    overlap_count = {cls: 0 for cls in classes}

    # Start counting samples from the most specific overlap region (highest number of classes)
    for class_comb, min_overlap, max_overlap in sorted(overlap_regions, key=lambda x: -len(x[0])):
        in_overlap = np.all((samples >= min_overlap) & (samples <= max_overlap), axis=1) & ~counted_samples

        # Mark these samples as counted
        counted_samples[in_overlap] = True

        # Count samples for each class in this overlapping region
        for cls in class_comb:
            count_cls = np.sum(in_overlap & (class_labels == cls))
            overlap_count[cls] += count_cls

            # Record counts for each overlap region for debugging
            region_key = '-'.join(map(str, class_comb))
            if region_key not in samples_in_overlap:
                samples_in_overlap[region_key] = {}
            samples_in_overlap[region_key][cls] = count_cls

    # Calculate Ambiguity
    total_samples_per_class = df[class_column].value_counts().to_dict()

    ambiguity_values = {cls: 0 for cls in classes}

    # Calculate ambiguity values
    for cls in classes:
        if total_samples_per_class[cls] > 0:
            ambiguity_values[cls] = overlap_count[cls] / total_samples_per_class[cls]

    mean_ambiguity = np.mean(list(ambiguity_values.values())) if ambiguity_values else 0.0

    # overlap_regions, samples_in_overlap can be used for debugging

    return mean_ambiguity


def calculate_continuous_error(df, label_column, random_state= 42):
    """
    Calculate error for continuous featured datasets using Recatangular segmentation and calculating the probability of that segment belong to the other class.
    Args:
        df (pandas dataframe): Dataset
        class_column (string): class column name
    returns:
        float: error for conitnuous case
    """
    # Train the decision tree
    feature_columns = [col for col in df.columns if col != label_column]
    X = df[feature_columns]
    y = df[label_column]

    clf = DecisionTreeClassifier(max_depth=None, random_state=random_state)
    clf.fit(X, y)

    training_accuracy = clf.score(X, y)
    # print(f'Training Accuracy: {training_accuracy * 100:.2f}%')

    # Function to extract rectangles and labels from a trained decision tree
    def get_rectangles_from_tree(tree):
        left = tree.children_left
        right = tree.children_right
        threshold = tree.threshold
        feature = tree.feature
        value = tree.value
        
        def recurse(node, bounds):
            if feature[node] == _tree.TREE_UNDEFINED:
                # It's a leaf node
                leaf_label = np.argmax(value[node][0])
                return [(bounds, leaf_label)]
            
            new_bounds_left = [list(b) for b in bounds]
            new_bounds_right = [list(b) for b in bounds]
            
            feature_index = feature[node]
            threshold_value = threshold[node]
            
            new_bounds_left[feature_index][1] = threshold_value
            new_bounds_right[feature_index][0] = threshold_value
            
            left_rectangles = recurse(left[node], new_bounds_left)
            right_rectangles = recurse(right[node], new_bounds_right)
            
            return left_rectangles + right_rectangles

        # Initialize bounds for each feature
        initial_bounds = [[-np.inf, np.inf] for _ in range(tree.n_features)]
        rectangles = recurse(0, initial_bounds)
        return rectangles

    # Extract rectangles and labels from the decision tree
    rectangles = get_rectangles_from_tree(clf.tree_)
    
    # Calculate KDE for each class complement
    # print(rectangles)
    classes = np.unique(df[label_column])
    # print(classes)
    kde_by_class = {}

    for cls in classes:
        class_data = df[df[label_column] != cls][feature_columns]
        kde_by_class[cls] = stats.gaussian_kde(class_data.T)

    # Calculate probabilities for the segments
    segment_probabilities = []
    for rect, predicted_label in rectangles:
        bounds_min = [b[0] for b in rect]
        bounds_max = [b[1] for b in rect]
        segment = df[np.all((df[feature_columns] >= bounds_min) & (df[feature_columns] < bounds_max), axis=1)]

        # print(f"Predicted label: {predicted_label}")
        # print(f"Available KDE class labels: {list(kde_by_class.keys())}")
        if predicted_label not in kde_by_class: # when there is not any sample in the segmented ragion decision tree assign a random label
            print(f"Warning: Predicted label '{predicted_label}' is not a valid class.")
            continue
        if not segment.empty:
            # Calculate the probability of misclassification for this segment
            kde = kde_by_class[predicted_label]
            error_probability = kde.integrate_box(bounds_min, bounds_max, maxpts=500000)
            
            segment_probabilities.append(error_probability)


    # Compute total error probability
    total_error_probability_all_segments = np.sum(segment_probabilities)
    return total_error_probability_all_segments