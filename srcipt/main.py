import sys
import os
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_generator import generate_synthetic_dataset
from src.visualization import data_visualization
from src.model import select_model

def main():
    df = generate_synthetic_dataset(1000)
    data_visualization(df)
    X = df[['feature']]
    y = df['Label']
    
    output_dir = 'output'
    output_file = os.path.join(output_dir, 'model_performance.txt')

    model_names = ["logistic_regression", "naive_bayes", "random_forest"]
        
    for model_name in model_names:
        model = select_model(model_name)
        model.fit(X, y)
        y_pred = model.predict(X)
        # Global performance
        global_accuracy = accuracy_score(y, y_pred)
        # Cross validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf)
        print(f"Model name: {model_name}, global_accuracy: {global_accuracy}, Scores for each fold: {scores}, Average_score: {np.mean(scores):.4f}")
        with open(output_file, 'a') as f:
            f.write(f"Model name: {model_name}, global_accuracy: {global_accuracy}, Scores for each fold: {scores}, Average_score: {np.mean(scores):.4f}\n")

if __name__ == "__main__":
    main()