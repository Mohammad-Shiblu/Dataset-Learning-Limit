import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_generator import generate_synthetic_dataset
from src.visualization import data_visualization
def main():
    df = generate_synthetic_dataset(1000)
    data_visualization(df)
    print(df)



if __name__ == "__main__":
    main()