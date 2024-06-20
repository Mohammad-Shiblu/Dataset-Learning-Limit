from scipy import stats
import numpy as np

def calculate_pdf(df):
    """Calculate pdf using kernel density estimate and return KDE obeject"""

    class_0_data = df[df["Label"] == 0]["feature"]
    class_1_data = df[df['Label'] == 1]['feature']
    kde0 = stats.gaussian_kde(class_0_data)
    kde1 = stats.gaussian_kde(class_1_data)

    return kde0, kde1

def calculate_cdf(kde, th):
    """Calcualte the cdf from -inf to certain threshold"""

    return kde.integrate_box_1d(-np.inf, th)

def calculate_crossover_area(cdf1, cdf2):
    """Calculate the common region between two pdf"""

    common_area = 1- cdf1  + cdf2
    return common_area

def calculate_theorotical_error(df):
    """calculate the theretical error"""

    kde0, kde1 = calculate_pdf(df)
    x = np.linspace(-15, 30, 50000)
    y1 = kde0.pdf(x)
    y2 = kde1.pdf(x)
    intersection_point = np.argmin(np.abs(y1-y2))

    cdf1 = calculate_cdf(kde0, x[intersection_point])
    cdf2 = calculate_cdf(kde1, x[intersection_point])
    area = calculate_crossover_area(cdf1, cdf2)
    return x[intersection_point], area

def calculate_maximum_coverage(df):
    """Calculate maximum coverage using """
    df = df.sort_values("feature")
    diff = df["Label"].diff()
    diff =diff.bfill()

    change_points = df["feature"][(np.abs(diff)>0).shift(-1, fill_value=False)].values + (df["feature"][np.abs(diff)>0].values - df["feature"][(np.abs(diff)>0).shift(-1, fill_value=False)].values) /2
    segments = [(df['feature'].min(), change_points[0])] + \
            [(change_points[i], change_points[i + 1]) for i in range(len(change_points) - 1)] + \
            [(change_points[-1], df['feature'].max())]
    segments

    class_0_data = df[df["Label"] == 0]["feature"]
    class_1_data = df[df["Label"] == 1]["feature"]

    kde_class_0 = stats.gaussian_kde(class_0_data)
    kde_class_1 = stats.gaussian_kde(class_1_data)
    
    segment_probabilities = []
    for start, end in segments:
        segment = df[(df['feature'] >= start) & (df['feature'] < end)]
        actual_value = segment["Label"].iloc[0]
        if actual_value == 0.0:
            segment_probabilities.append(kde_class_1.integrate_box_1d(start, end))
        else:
            segment_probabilities.append(kde_class_0.integrate_box_1d(start, end))
            
    total_error_probability_all_segments = np.sum(segment_probabilities)
    return total_error_probability_all_segments
