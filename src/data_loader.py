import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo 

class DataLoader:
    def __init__(self):
        self.dataset_loaders = {
            "continuous": {
                "iris": self.load_iris,
                "kidney_stone": self.load_kidney_stone,
                "wine": self.load_wine,
                "blood_transfusion": self.load_blood_transfusion,
                "rice": self.load_rice,

            },
            "discrete": {
                'car': self.load_car,
                'zoo': self.load_zoo,
                'tictac': self.load_tictac,
                'balance': self.load_balance,
                'lens':self.load_lens,
            },
            "mixed": {
                # Add mixed datasets here
                'anaemia': self.load_anaemia,
                'ecoli': self.load_ecoli,
                'social_network': self.load_social_network,
            }
        }

    def get_available_datasets(self):
        available_datasets = {}
        for data_type, datasets in self.dataset_loaders.items():
            available_datasets[data_type] = list(datasets.keys())
        return available_datasets
    
    def select_dataset(self, data_type, dataset_name):
        if data_type in self.dataset_loaders and dataset_name in self.dataset_loaders[data_type]:
            return self.dataset_loaders[data_type][dataset_name]()
        else:
            raise ValueError(f"Dataset {dataset_name} of type {data_type} is not supported.")
    
    def load_iris(self):
        iris = load_iris()
        iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        iris_df['class'] = iris.target_names[iris.target]
        label_encoder = LabelEncoder()
        iris_df['class'] = label_encoder.fit_transform(iris_df['class'])
        return iris_df

    def load_kidney_stone(self):
        df = pd.read_csv("../data/continuous/kidney-stone-dataset.csv")
        kidney_stone_df = df.drop("Unnamed: 0", axis=1)
        kidney_stone_df.rename(columns={'target': 'class'}, inplace=True)
        return kidney_stone_df
    
    def load_wine(self): 
        df_wine = pd.read_csv("../data/continuous/winequality-red.csv", sep= ";")
        df_wine.rename(columns={'quality': 'class'}, inplace=True)
        label_encoder = LabelEncoder()
        df_wine['class'] = label_encoder.fit_transform(df_wine['class'])
        return df_wine
    # probelm with error calculation as decision can't be trained to 100 percent
    def load_blood_transfusion(self):
        blood_transfusion_service_center = fetch_ucirepo(id=176) 
        X = blood_transfusion_service_center.data.features 
        y = blood_transfusion_service_center.data.targets 
        blood_df = pd.concat([X, y], axis=1)
        blood_df.rename(columns={'Donated_Blood': 'class'}, inplace=True)
        return blood_df
    
    def load_rice(self):
        rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 
        X = rice_cammeo_and_osmancik.data.features 
        y = rice_cammeo_and_osmancik.data.targets 
        rice_df = pd.concat([X, y], axis=1)
        label_encoder = LabelEncoder()
        rice_df['Class'] = label_encoder.fit_transform(rice_df['Class'])
        rice_df.rename(columns={'Class': 'class'}, inplace=True)
        return rice_df

    def load_anaemia(self):
        df_anaemia = pd.read_csv("../data/continuous/anaemia.csv")
        label_encoder = LabelEncoder()
        df_anaemia['Anaemic'] = label_encoder.fit_transform(df_anaemia['Anaemic'])
        df_anaemia = df_anaemia.drop(columns=['Number'])
        return df_anaemia
    
    def load_social_network(self):
        network_df = pd.read_csv('../data/mixed/Social_Network_Ads.csv')
        return network_df
     
    def load_car(self):
        car_df = pd.read_csv('../data/discrete/car.csv')
        label_encoder = LabelEncoder()
        for column in car_df.columns:
            car_df[column] = label_encoder.fit_transform(car_df[column])
        return car_df
    
    def load_tictac(self):
        tic_tac_toe_endgame = fetch_ucirepo(id=101)  
        X = tic_tac_toe_endgame.data.features 
        y = tic_tac_toe_endgame.data.targets 
        tictac_df = pd.concat([X, y], axis=1)
        label_encoder = LabelEncoder()
        for column in tictac_df.columns:
            if tictac_df[column].dtype == 'object':  
                tictac_df[column] = label_encoder.fit_transform(tictac_df[column])
        return tictac_df
    
    def load_nursery(self):
        nursery = fetch_ucirepo(id=76) 
        X = nursery.data.features 
        y = nursery.data.targets 
        nursery_df = pd.concat([X, y], axis=1)
        label_encoder = LabelEncoder()
        for column in nursery_df.columns:
            if nursery_df[column].dtype == 'object':  
                nursery_df[column] = label_encoder.fit_transform(nursery_df[column])
        return nursery_df
    
    def load_ecoli(self):
        ecoli = fetch_ucirepo(id=39)
        X = ecoli.data.features 
        y = ecoli.data.targets 
        # Combine features and target into a single DataFrame (optional, just for inspection)
        ecoli_df = pd.concat([X, y], axis=1)
        label_encoder = LabelEncoder()
        ecoli_df['class'] = label_encoder.fit_transform(ecoli_df['class'])
        return ecoli_df
    
    def load_balance(self):
        balance_scale = fetch_ucirepo(id=12) 
        X = balance_scale.data.features 
        y = balance_scale.data.targets 
        balance_df = pd.concat([X, y], axis=1)
        label_encoder = LabelEncoder()
        balance_df['class'] = label_encoder.fit_transform(balance_df['class'])
        return balance_df
    
    def load_lens(self):
        lenses = fetch_ucirepo(id=58) 
        X = lenses.data.features 
        y = lenses.data.targets 
        lenses_df = pd.concat([X, y], axis=1)
        return lenses_df

    def load_zoo(self): 
        zoo = fetch_ucirepo(id=111) 
        X = zoo.data.features 
        y = zoo.data.targets 
        zoo_df = pd.concat([X, y], axis=1)
        return zoo_df