import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_generator import generate_synthetic_dataset

def main():
    df = generate_synthetic_dataset(1000)
    print(df)



if __name__ == "__main__":
    main()