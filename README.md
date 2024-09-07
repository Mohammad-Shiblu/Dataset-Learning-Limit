# DLL_Seminar_SS_2024
## Ambiguity in Datasets
Ambiguity in data presents a substantial challenge in the classification process, significantly limiting the effectiveness of classification models. Ambiguity often arises in weakly structured datasets where a significant portion of data samples dispersed around the whole feature space. The following shows a case for ambiguity for continuous and discrete cases. 

<div style="text-align: center;">
  <img src="output/ambiguity.png" alt="Sample Figure" width="500" style="border: 1px solid #ddd; border-radius: 8px; padding: 10px;">
</div>

In this project we defined the the ambiguity and derived mathematical expressions for ambiguity and error for continuous and discrete datasets. Additionally, we discuss the merits and the potential limitations of our methods by implementing it on different real world datasets.

To minimize complications, we adopted a case-by-case approach by considering ambiguity and potential errors specific to each situation. As a result, we divided the concept into two distinct types of datasets: those with continuous features and those with discrete features. 

In datasets with continuous features, values can range across the entire set of real numbers. Ambiguity in this context arises from the overlap between feature spaces of different classes. Accurate estimation of this ambiguity requires knowledge of the distribution of features within each class. However, since domain knowledge about the distribution of real-world datasets is often unavailable, we must generalize our approach.

To generalize, we use a rectangular hypercube to represent the feature space. Ambiguity arises from data samples located within the overlapping regions of these hypercubes, which represent different classes. This ambiguity can be quantified using the following equation:
$$  A_c = \frac{1}{K} \sum_{j}^{K} \frac{{|x_i \in C_j \cap C_{K \setminus j}}|}{|x_i \in C_j|} $$

Here, $|x_i \in C_j \cap C_{K \setminus C_j}|$  denotes the number of samples that belong to class $j$ and also reside in the overlapping regions with other classes., while $|x_i \in  C_j|$ represents the number of samples belonging to class j. k denotes the total number of classes.
<div style="text-align: center;">
  <img src="output/hypercubes.png" alt="Sample Figure" width="500" style="border: 1px solid #ddd; border-radius: 8px; padding: 10px;">
</div>
The figure shows an example of hypercubes on synthetically generated dataset.
<br>
The maximum error resulting from this ambiguity can be estimated by greedily segmenting the dataset such that each segment ${R_k}$ , contains only the samples of class k. Within these segments, we then estimate the probability that the samples belong to classes other than their original class k. This concept can be mathematically formulated as follows:

$$E_c = \sum_{1}^{S} \int_{R_k} P(R_k \notin k)$$

Here, $E_c$ denotes the error for the continuous case, while $S$ represents the total number of segments. ${R_k}$ refers to an individual segmented hypercube, and 
$P(R_k\notin k)$  indicates the probability that the samples within this segment belong to classes other than the original class k.

<div style="text-align: center;">
  <img src="output/segmentation.png" alt="Sample Figure" width="500" style="border: 1px solid #ddd; border-radius: 8px; padding: 10px;">
</div>
The figure above shows the segmentation process achieve on syntehtic datasets using decision tree classifier.

## Installation
Step 1:  To run this project, you need to have Python installed along with the necessary dependencies. You can install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

Step 2: clone the repository.
```bash
git https://gitlab.cs.fau.de/cdh-seminars/dataset-learning-limit/dll_seminar_ss_2024.git
```
step 3: Execute the main Python script in script to start the application:
```bash
python main.py
```
step 4: Interact with the GUI:
* Choose a dataset type from the dropdown menu.
* Select a dataset for analysis 
* View the results directly in the app.

## App visual
Below is the screenshot of the application interface:
<div style="text-align: center;">
  <img src="output/window.png" alt="Sample Figure" width="500" style="border: 1px solid #ddd; border-radius: 8px; padding: 10px;">
</div>
