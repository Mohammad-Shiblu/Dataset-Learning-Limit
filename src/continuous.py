import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier, export_text, _tree
from scipy import stats
from scipy.spatial import ConvexHull
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

class ContinuousAmbiguityError:
    def __init__(self, X=None, y=None):
        self.X = X
        self.y = y
        self.random_state = 42

    def calculate_ambiguity_hypercube(self, X=None, y=None):
        
        self.X = X if X is not None else self.X
        self.y = y if y is not None else self.y
        features = np.arange(self.X.shape[1])  # Feature indices
        classes = np.unique(self.y)  # Unique class labels

        # Calculate Hypercubes
        hypercubes = {}
        for cls in classes:
            class_indices = np.where(self.y == cls)[0]
            class_samples = self.X[class_indices]
            min_values = np.min(class_samples, axis=0)
            max_values = np.max(class_samples, axis=0)
            hypercubes[cls] = (min_values, max_values)

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
        samples = self.X
        class_labels = self.y

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
        total_samples_per_class = {cls: np.sum(class_labels == cls) for cls in classes}

        ambiguity_values = {cls: 0 for cls in classes}

        # Calculate ambiguity values
        for cls in classes:
            if total_samples_per_class[cls] > 0:
                ambiguity_values[cls] = overlap_count[cls] / total_samples_per_class[cls]

        mean_ambiguity = np.mean(list(ambiguity_values.values())) if ambiguity_values else 0.0

        # overlap_regions, samples_in_overlap can be used for debugging

        return mean_ambiguity, overlap_regions, samples_in_overlap

    def calculate_ambiguity_convex_hull(self, X=None, y=None):
        self.X = X if X is not None else self.X
        self.y = y if y is not None else self.y
        classes = np.unique(self.y)
        K = len(classes)                 # total number of classes
        n_features = self.X.shape[1]
        class_hulls = {}

        for cls in classes:
            class_points = self.X[self.y==cls]
            if len(class_points) < n_features:
                raise ValueError(f"class {cls} has insufficient points for convex hull")
            class_hulls[cls] = ConvexHull(class_points)

        def points_in_hull(points, hull):
            coef = hull.equations[:, :-1]
            const = hull.equations[:, -1]
            return np.all(np.dot(points, coef.T) + const <= 0, axis=1)
        
        total_ambiguity = 0
        for cls_j in classes:
            class_j_points = self.X[self.y== cls_j]
            total_points_j = len(class_j_points)
            overlap_count_j = 0
            for cls_k in classes:
                if cls_k != cls_j:
                    overlap_points = class_j_points[points_in_hull(class_j_points, class_hulls[cls_k])]
                    overlap_count_j += len(overlap_points)

            if total_points_j > 0:
                total_ambiguity += overlap_count_j / total_points_j
        ambiguity = total_ambiguity / K
        return ambiguity


    def calculate_continuous_error(self, X=None, y=None):
        
        self.X = X if X is not None else self.X
        self.y = y if y is not None else self.y
        # Train the decision tree
        # feature_columns = [col for col in self.df.columns if col != self.class_column]
        # X = self.df[feature_columns]
        # y = self.df[self.class_column]
        clf = DecisionTreeClassifier(max_depth=None, random_state=self.random_state)
        clf.fit(self.X, self.y)
        training_accuracy = clf.score(self.X, self.y)
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
        classes = np.unique(self.y)
        # print(classes)
        kde_by_class = {}

        for cls in classes:
            class_data = self.X[self.y != cls]
            kde_by_class[cls] = stats.gaussian_kde(class_data.T)

        # Calculate probabilities for the segments
        segment_probabilities = []
        for rect, predicted_label in rectangles:
            bounds_min = [b[0] for b in rect]
            bounds_max = [b[1] for b in rect]
            in_segment = np.all((self.X >= bounds_min) & (self.X < bounds_max), axis=1)
            segment = self.X[in_segment]

            if segment.shape[0] == 0:
                continue

            # print(f"Predicted label: {predicted_label}")
            # print(f"Available KDE class labels: {list(kde_by_class.keys())}")
            if predicted_label not in kde_by_class: # when there is not any sample in the segmented ragion decision tree assign a random label
                print(f"Warning: Predicted label '{predicted_label}' is not a valid class.")
                continue
            
                # Calculate the probability of misclassification for this segment
            kde = kde_by_class[predicted_label]
            error_probability = kde.integrate_box(bounds_min, bounds_max, maxpts=500000)
            segment_probabilities.append(error_probability)

        # Compute total error probability
        total_error_probability_all_segments = np.sum(segment_probabilities)
        return total_error_probability_all_segments

    def calculate_accuracy_limit(self,X=None, y=None):
        self.X = X if X is not None else self.X
        self.y = y if y is not None else self.y
        
        # Step 1: Prepare classes and priors
        classes = np.unique(self.y)
        n_classes = len(classes)
        n_samples = self.X.shape[0]
        priors = [np.sum(self.y == c) / len(self.y) for c in classes]
        
        # Step 2: Estimate p_lea(x|i) using KDE
        kde_models = {}
        for c in classes:
            kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
            kde.fit(self.X[self.y == c])
            kde_models[c] = kde
        
        # Step 3: Compute p_lea(x|i) and qcla(x|j)
        p_lea = np.zeros((n_samples, n_classes))  
        for i, c in enumerate(classes):
            log_density = kde_models[c].score_samples(self.X)
            p_lea[:, i] = np.exp(log_density)
        
        # Compute qcla(x|j)
        numerator = p_lea * priors
        denominator = np.sum(numerator, axis=1, keepdims=True)
        qcla = numerator / denominator
        
        # Step 4: Binary classification decision (class indicator function)
        qcla_binary = np.zeros_like(qcla)
        qcla_binary[np.arange(n_samples), np.argmax(qcla, axis=1)] = 1
        
        # Step 5: Compute confusion matrix
        confusion_matrix = np.zeros((n_classes, n_classes))
        for j in range(n_classes):
            for i in range(n_classes):
                confusion_matrix[j, i] = np.sum(qcla_binary[:, j] * p_lea[:, i]) / n_samples
        
        # Step 6: Compute accuracy limit (Amax)
        Amax = np.sum([priors[i] * confusion_matrix[i, i] for i in range(n_classes)])
        
        return Amax, confusion_matrix
    
    def theretical_accuracy_limit(self, X=None, y=None, bandwidth = 0.2):
        self.X = X if X is not None else self.X
        self.y = y if y is not None else self.y
        classes = np.unique(self.y)
        n_classes = len(classes)
        class_counts = np.bincount(self.y)
        class_priors = class_counts / len(self.y)
        log_class_priors = np.log(class_priors)
        kdes = []
        for c in classes:
            X_c = self.X[self.y == c]
            kde = KernelDensity(bandwidth = bandwidth)
            kde.fit(X_c)
            kdes.append(kde)

        log_likelihood = np.zeros((len(self.X), n_classes))
        for i, kde in enumerate(kdes):
            log_likelihood[:, i] = kde.score_samples(self.X)

        log_posteriors = log_likelihood + log_class_priors
        y_pred = np.argmax(log_posteriors, axis=1)
        cm = confusion_matrix(self.y, y_pred)
        accuracy = np.trace(cm) / np.sum(cm)
        return accuracy
