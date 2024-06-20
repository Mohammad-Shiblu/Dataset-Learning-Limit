import numpy as np
import pandas as pd


def generate_synthetic_dataset(num_samples):
    """ Takes num of samples and return the df with one features and two class """
    
    np.random.seed(0)
    samples_per_class = int(num_samples /2)
    label_0 = np.zeros(samples_per_class)
    label_1 = np.ones(samples_per_class)

    # class 1 data
    mu_1 = 12
    sigma_1 = 5
    class_1_data = np.random.normal(mu_1, sigma_1, samples_per_class)

    # class 0 data
    mu_0 = 0
    sigma_0 = 4
    class_0_data = np.random.normal(mu_0, sigma_0, samples_per_class)
    data  =  np.concatenate((class_0_data, class_1_data)) 
    labels = np.concatenate((label_0, label_1))

    d = {
        "feature": data,
        "Label": labels
    }
    df = pd.DataFrame(data =d)
    return df