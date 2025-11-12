import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo 

class DataLoader:
    def __init__(self):
        self.dataset_loaders = {
            "Iris": self.load_iris,
            "Kidney Stone": self.load_kidney_stone,
            "Wine": self.load_wine,
            "Blood Transfusion": self.load_blood_transfusion,
            "Anaemia": self.load_anaemia,
            "Breast Cancer": self.load_breast_cancer,
            "Rice": self.load_rice,
            "gender Classification": self.load_gender_classification,
            "Dry Bean": self.load_dry_bean,
            "Heart Attack": self.load_heart_attack
            
        }

    def get_available_datasets(self):
        """Returns a list of all available dataset names"""
        return list(self.dataset_loaders.keys())
    
    def load_dataset(self, dataset_name):
        """Load a dataset by name"""
        if dataset_name in self.dataset_loaders:
            return self.dataset_loaders[dataset_name]()
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported.")
    
    # dataset 1
    def load_iris(self):
        iris = load_iris()
        iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        iris_df['class'] = iris.target_names[iris.target]
        label_encoder = LabelEncoder()
        iris_df['class'] = label_encoder.fit_transform(iris_df['class'])
        return iris_df

    # dataset 2
    def load_kidney_stone(self):
        df = pd.read_csv("./data/continuous/kidney-stone-dataset.csv")
        kidney_stone_df = df.drop("Unnamed: 0", axis=1)
        kidney_stone_df.rename(columns={'target': 'class'}, inplace=True)
        return kidney_stone_df
    
    # dataset 3
    def load_wine(self): 
        df_wine = pd.read_csv("./data/continuous/winequality-red.csv", sep=";")
        df_wine.rename(columns={'quality': 'class'}, inplace=True)
        label_encoder = LabelEncoder()
        df_wine['class'] = label_encoder.fit_transform(df_wine['class'])
        return df_wine
    
    # dataset 4
    def load_blood_transfusion(self):
        blood_transfusion_service_center = fetch_ucirepo(id=176) 
        X = blood_transfusion_service_center.data.features 
        y = blood_transfusion_service_center.data.targets 
        blood_df = pd.concat([X, y], axis=1)
        blood_df.rename(columns={'Donated_Blood': 'class'}, inplace=True)
        return blood_df

    # dataset 5
    def load_anaemia(self):
        df_anaemia = pd.read_csv("./data/continuous/anaemia.csv")
        label_encoder = LabelEncoder()
        df_anaemia['Anaemic'] = label_encoder.fit_transform(df_anaemia['Anaemic'])
        df_anaemia = df_anaemia.drop(columns=['Number'])
        df_anaemia.rename(columns={'Anaemic': 'class'}, inplace=True)
        return df_anaemia
    
    # dataset 6
    def load_breast_cancer(self):
        breast_cancer = fetch_ucirepo(id=15) 
        X = breast_cancer.data.features 
        y = breast_cancer.data.targets 
        breast_df = pd.concat([X, y], axis=1)
        breast_df.rename(columns={'Class': 'class'}, inplace=True)
        return breast_df
    
    # dataset 7
    def load_rice(self):
        df_rice = pd.read_csv("./data/continuous/rice.csv")
        label_encoder = LabelEncoder()
        df_rice['Class'] = label_encoder.fit_transform(df_rice['Class'])
        df_rice.rename(columns={'Class': 'class'}, inplace=True)
        return df_rice
    
    # dataset 8
    def load_gender_classification(self):
        df_gender = pd.read_csv("./data/continuous/gender_classification_v7.csv")
        LabelEncoder = LabelEncoder()
        df_gender['gender'] = LabelEncoder.fit_transform(df_gender['gender'])
        df_gender.rename(columns={'gender': 'class'}, inplace=True)
        return df_gender

    # dataset 9
    def load_dry_bean(self):
        df_dry_bean = pd.read_csv("./data/continuous/Dry_Bean_Dataset.csv")
        label_encoder = LabelEncoder()
        df_dry_bean['Class'] = label_encoder.fit_transform(df_dry_bean['Class'])
        df_dry_bean.rename(columns={'Class': 'class'}, inplace=True)
        return df_dry_bean
    
    # dataset 10 
    def load_heart_attack(self):
        df_heart_attack = pd.read_csv("./data/continuous/Heart_Attack.csv")
        label_encoder = LabelEncoder()
        df_heart_attack['class'] = label_encoder.fit_transform(df_heart_attack['class'])
        return df_heart_attack
    
    # dataset 11
    # add more datasets as needed