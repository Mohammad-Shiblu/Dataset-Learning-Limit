import customtkinter as ctk
from tkinter import messagebox
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'src'))
if src_dir not in sys.path:
    sys.path.append(src_dir)

import continuous, discrete
from data_loader import DataLoader


def load_internal_dataset(dataset_type, dataset_name):
    dataset_type = dataset_type.lower()
    dataset_name = dataset_name.lower()
    loader = DataLoader()
    df = loader.select_dataset(dataset_type, dataset_name)
    return df

def calculate_and_display_results(df, dataset_type):
    class_name = df.columns[-1]  # Assuming last column is the class

    if dataset_type == 'Discrete':
        ambiguity = discrete.calculate_discrete_ambiguity(df, class_name)
        total_error_probability, _ = discrete.calculate_discrete_error(df, class_name)
    elif dataset_type == 'Continuous':
        ambiguity = continuous.calculate_continuous_ambiguity(df, class_name)
        total_error_probability = continuous.calculate_continuous_error(df, class_name)
    # elif dataset_type == 'mixed':
    #     ambiguity = model.calculate_mixed_ambiguity(df, class_name)
    #     total_error_probability, _ = model.calculate_mixed_error(df, class_name)
    else:
        messagebox.showerror("Error", "Invalid dataset type selected.")
        return
    
    result_text.set(f"Ambiguity: {ambiguity:.4f}\nError: {total_error_probability:.4f}")



class DatasetApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("600x400") 
        self.root.title("Dataset Analyzer")
        ctk.set_appearance_mode("Light")  
        ctk.set_default_color_theme("blue")  
        
        self.main_frame = ctk.CTkFrame(root)
        self.main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        ctk.CTkLabel(self.main_frame, text="Select Dataset Type:", font=("Arial", 15)).grid(row=0, column=0, padx=20, pady=20, sticky="w")
        self.dataset_type_var = ctk.StringVar(value="Continuous")
        dataset_types = ['Discrete', 'Continuous']  # Add mixed data set types in the future
        self.dataset_type_menu = ctk.CTkOptionMenu(self.main_frame, variable=self.dataset_type_var, values=dataset_types, command=self.update_dataset_menu, font=("Arial", 15))
        self.dataset_type_menu.grid(row=0, column=1, padx=20, pady=20, sticky="ew")

        ctk.CTkLabel(self.main_frame, text="Select Dataset:", font=("Arial", 15)).grid(row=1, column=0, padx=20, pady=20, sticky="w")
        self.dataset_var = ctk.StringVar(value="Lens")
        self.datasets = {
            'Discrete': ['Lens', 'Car', 'Zoo', 'Tictac', 'Balance'],
            'Continuous': ['Iris', 'Kidney_stone', 'Wine'],
        }
        self.dataset_menu = ctk.CTkOptionMenu(self.main_frame, variable=self.dataset_var, values=self.datasets['Continuous'], font=("Arial", 15))
        self.dataset_menu.grid(row=1, column=1, padx=20, pady=20, sticky="ew")

        # Button to dataset
        self.load_internal_button = ctk.CTkButton(self.main_frame, text="Load Dataset", command=self.load_dataset, font=("Arial", 15))
        self.load_internal_button.grid(row=2, column=0, columnspan=2, padx=20, pady=30)

        # Result display
        global result_text
        result_text = ctk.StringVar()
        self.result_label = ctk.CTkLabel(self.main_frame, textvariable=result_text, wraplength=400, justify="left", font=("Arial", 15))
        self.result_label.grid(row=3, column=0, columnspan=2, padx=20, pady=20)

    # Update the internal dataset menu based on selected dataset type
    def update_dataset_menu(self, *args):
        dataset_type = self.dataset_type_var.get()
        self.dataset_menu.configure(values=self.datasets[dataset_type])
        self.dataset_var.set(self.datasets[dataset_type][0])  # Set default value to the first dataset

    # Load Internal Dataset
    def load_dataset(self):
        dataset_type = self.dataset_type_var.get()
        dataset_name = self.dataset_var.get()
        try:
            df = load_internal_dataset(dataset_type, dataset_name)
            calculate_and_display_results(df, dataset_type)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load internal dataset: {str(e)}")


if __name__ == "__main__":
    root = ctk.CTk()  
    app = DatasetApp(root)
    root.mainloop()
