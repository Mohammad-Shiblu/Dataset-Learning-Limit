from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, KFold
from src.continuous import calc_ambiguity_np

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
            self.X_train, self.y_train = X_train, y_train
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

        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy
    
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
            train_ambiguity = calc_ambiguity_np(X_train, y_train)  # Pass both X and y
            train_predictions = self.model.predict(X_train)
            train_error = sum(train_predictions != y_train) / len(y_train)

            # Calculate Test Ambiguity and Error
            test_ambiguity = calc_ambiguity_np(X_test, y_test)  # Pass both X and y
            test_predictions = self.model.predict(X_test)
            test_error = sum(test_predictions != y_test) / len(y_test)

            # Evaluate accuracy
            train_accuracy = self.model.score(X_train, y_train)
            test_accuracy = self.model.score(X_test, y_test)

            # Store results
            results.append({
                'fold': fold + 1,
                'train_ambiguity': train_ambiguity,
                'train_error': train_error,
                'train_accuracy': train_accuracy,
                'test_ambiguity': test_ambiguity,
                'test_error': test_error,
                'test_accuracy': test_accuracy
            })

        return results
    
