import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def data_visualization(df):
    class_0_data = df[df['Label'] == 0]['feature']
    class_1_data = df[df['Label'] == 1]['feature']

    plt.figure()
    sns.kdeplot(class_0_data, color='blue', label='Class 0')
    sns.kdeplot(class_1_data, color='red', label='Class 1')

    plt.scatter(class_0_data, np.zeros_like(class_0_data), color='blue', alpha=0.6, label='Class 0 Samples')
    plt.scatter(class_1_data, np.zeros_like(class_1_data), color='red', alpha=0.6, label='Class 1 Samples')

    plt.title('Class Conditional Density Plot of Feature')
    plt.ylim(-0.05, .15)
    plt.xlabel('Feature')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    
    file_name = "kde_plot.png"
    output_dir = "output"
    output_path = os.path.join(output_dir, file_name)
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")
