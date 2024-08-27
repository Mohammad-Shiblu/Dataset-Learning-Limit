import pandas as pd
import numpy as np
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

    # Calculate Overlap Regions
    overlap_regions = []
    class_list = list(hypercubes.keys())
    num_classes = len(class_list)
    
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            min_overlap = np.maximum(hypercubes[class_list[i]][0], hypercubes[class_list[j]][0])
            max_overlap = np.minimum(hypercubes[class_list[i]][1], hypercubes[class_list[j]][1])
            if np.all(min_overlap <= max_overlap):
                overlap_regions.append((class_list[i], class_list[j], min_overlap, max_overlap))

    # Count Samples in Overlap Regions
    samples = df[features].values
    class_labels = df[class_column].values

    samples_in_overlap = {}

    for cls1, cls2, min_overlap, max_overlap in overlap_regions:
        in_overlap_cls1 = np.all((samples >= min_overlap) & (samples <= max_overlap), axis=1) & (class_labels == cls1)
        in_overlap_cls2 = np.all((samples >= min_overlap) & (samples <= max_overlap), axis=1) & (class_labels == cls2)

        count_cls1 = np.sum(in_overlap_cls1)
        count_cls2 = np.sum(in_overlap_cls2)

        region_key = f'{cls1}-{cls2}'
        samples_in_overlap[region_key] = {cls1: count_cls1, cls2: count_cls2} # samples in the overlap region

    # Calculate Ambiguity
    total_samples_per_class = df[class_column].value_counts().to_dict()
    # print("total samples per class:")
    # print(total_samples_per_class)
    ambiguity_values = {cls: 0 for cls in classes}

    for region, counts in samples_in_overlap.items():
        cls1, cls2 = region.split('-')
        cls1 = float(cls1)  # Ensure class labels are correctly interpreted
        cls2 = float(cls2)
        if cls1 in total_samples_per_class and cls2 in total_samples_per_class:
            if total_samples_per_class[cls1] > 0:
                ambiguity_values[cls1] += counts[cls1] / total_samples_per_class[cls1]
            if total_samples_per_class[cls2] > 0:
                ambiguity_values[cls2] += counts[cls2] / total_samples_per_class[cls2]

    mean_ambiguity = np.mean(list(ambiguity_values.values())) if ambiguity_values else 0.0
    
    # overlap_regions, samples_in_overlap can be used to check the overlalped regions and the samples number in the overlapped regions.

    return mean_ambiguity


def calculate_continuous_error(df, label_column):
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

    clf = DecisionTreeClassifier(max_depth=None)
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
    # print(rectangles)
    # Calculate KDE for each class complement
    classes = np.unique(df[label_column])
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

        if not segment.empty:
            # Calculate the probability of misclassification for this segment
            kde = kde_by_class[predicted_label]
            error_probability = kde.integrate_box(bounds_min, bounds_max, maxpts=500000)
            
            segment_probabilities.append(error_probability)


    # Compute total error probability
    total_error_probability_all_segments = np.sum(segment_probabilities)
    return total_error_probability_all_segments