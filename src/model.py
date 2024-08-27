from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, cross_val_score

class Model:
    def __init__(self, model_type = 'rf', **kwargs):
        self.model_name = model_type
        self.model = self.initialize_model(model_type, **kwargs)
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

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
        X= df[feature_col]
        y = df[class_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    
    def cross_validate(self, cv =5):
        scores = cross_val_score(self.model, self.X_train, self.y_train, cv=cv) 
        return scores.mean(), scores