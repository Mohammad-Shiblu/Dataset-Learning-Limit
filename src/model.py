from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from continuous import ContinuousAmbiguityError
import numpy as np
from sklearn.metrics import balanced_accuracy_score

class Model:
    def __init__(self, model_type = 'rf', **kwargs):
        self.model_name = model_type
        self.model = self.initialize_model(model_type, **kwargs)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X = None
        self.y = None

    def initialize_model(self, model_type, **kwargs):
        models = {
            'rf': RandomForestClassifier,
            'svm': SVC,
            'lr': LogisticRegression,
            'nb': GaussianNB
        }
        if model_type in models:
            return models[model_type](**kwargs)
        else:
            raise ValueError(f"Model type {model_type} not supported. Supported models are: 'rf', 'svm', 'lr, 'nb'")
        
    def data_split(self, df, test_size = .2, random_state = 42):
        feature_col = df.columns[:-1]
        class_col = df.columns[-1]
        self.X= df[feature_col].values
        self.y = df[class_col].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train(self, df=None, X_train=None, y_train=None, test_size=0.2, random_state=42):
        if df is not None:
            self.data_split(df, test_size, random_state)
        elif X_train is not None and y_train is not None:
            self.X = X_train
            self.y = y_train
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        else:
            raise ValueError("Either a DataFrame or X_train and y_train must be provided.")

        self.model.fit(self.X_train, self.y_train)

    def predict(self, X_test=None):
        if X_test is None:
            X_test = self.X_test
        return self.model.predict(X_test)
    
    def evaluate(self, X_test=None, y_test=None):
        if X_test is None or y_test is None:
            X_test, y_test = self.X_test, self.y_test

        test_predictions = self.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)

        train_predictions = self.predict(self.X_train)
        train_accuracy = accuracy_score(self.y_train, train_predictions)

        return train_accuracy, test_accuracy
    
    def class_base_accuracy(self, y_test, y_pred):
        classes = np.unique(y_test)
        accuracy_values = []
        for cls in classes:
            indices = (y_test == cls)
            correct_predictions = (y_pred[indices] == y_test[indices]).sum()
            total_samples = indices.sum()
            acc = correct_predictions / total_samples if total_samples > 0 else 0
            accuracy_values.append(acc)

        avg_class_accuracy = np.mean(accuracy_values)
        return avg_class_accuracy

    
    def cross_validate_with_metrics(self, cv=5):
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        results = []

        for fold, (train_index, test_index) in enumerate(kf.split(self.X)):
            # Use .iloc if self.X and self.y are pandas DataFrames
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # Train model
            self.model.fit(X_train, y_train)

            # Calculate Train Ambiguity and Error
            train = ContinuousAmbiguityError(X_train, y_train)
            train_ambiguity = train.calculate_ambiguity_convex_hull()
            train_predictions = self.model.predict(X_train)
            train_error = sum(train_predictions != y_train) / len(y_train)
            train_class_accuracy = balanced_accuracy_score(y_train, train_predictions)
            train_theretical_limit = train.theretical_accuracy_limit()
            train_theretical_error = train.calculate_continuous_error()

            # Calculate Test Ambiguity and Error
            test = ContinuousAmbiguityError(X_test, y_test)
            test_ambiguity = test.calculate_ambiguity_convex_hull()  # Pass both X and y
            test_predictions = self.model.predict(X_test)
            test_error = sum(test_predictions != y_test) / len(y_test)
            test_class_accuracy = balanced_accuracy_score(y_test, test_predictions)
            test_theretical_limit = test.theretical_accuracy_limit()
            test_theretical_error = test.calculate_continuous_error()

            # Evaluate accuracy
            train_accuracy = self.model.score(X_train, y_train)
            test_accuracy = self.model.score(X_test, y_test)

            # Store results
            results.append({
                'fold': fold + 1,
                'train_ambiguity': train_ambiguity,
                'train_error': train_error,
                'train_accuracy': train_accuracy,
                'train_balanced_accuracy': train_class_accuracy,
                'train_theretical_limit' : train_theretical_limit,
                'train_theretical_error' : train_theretical_error,
                'test_ambiguity': test_ambiguity,
                'test_error': test_error,
                'test_accuracy': test_accuracy,
                'test_balanced_accuracy': test_class_accuracy,
                'test_theretical_limit' : test_theretical_limit,
                'test_theretical_error' : test_theretical_error,
            })

        return results
    
    
