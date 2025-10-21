import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

# Importing DataLoader class
from src.data_loader import DataLoader  

# This function segments the feature space using a decision tree
def segmentation(X, y, verbose=True):
    clf = DecisionTreeClassifier(max_depth=None, random_state=42)
    clf.fit(X, y)
    training_accuracy = clf.score(X, y)
    if verbose:
        st.write(f"Decision Tree training accuracy: {training_accuracy * 100:.2f}%")

    def get_rectangles_from_tree(tree):
        left = tree.children_left
        right = tree.children_right
        threshold = tree.threshold
        feature = tree.feature
        value = tree.value

        def recurse(node, bounds):
            if feature[node] == _tree.TREE_UNDEFINED:
                leaf_label = np.argmax(value[node][0])
                return [(bounds, leaf_label)]
            
            new_bounds_left = [list(b) for b in bounds]
            new_bounds_right = [list(b) for b in bounds]
            
            feature_index = feature[node]
            threshold_value = threshold[node]
            
            new_bounds_left[feature_index][1] = threshold_value
            new_bounds_right[feature_index][0] = threshold_value
            
            left_rectangles = recurse(left[node], new_bounds_left)
            right_rectangles = recurse(right[node], new_bounds_right)
            
            return left_rectangles + right_rectangles

        initial_bounds = [[-np.inf, np.inf] for _ in range(tree.n_features)]
        rectangles = recurse(0, initial_bounds)
        return rectangles

    rectangles = get_rectangles_from_tree(clf.tree_)
    if verbose:
        st.write(f"Total mini clusters (rectangles) extracted: {len(rectangles)}")
    return rectangles

# calculates overfitting metrics only on test-only segments that does not contain training samples
def is_overfitting(df, classifier, test_size=0.2, random_state=42, verbose=True):
    X = df.drop(columns=['class'])
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    classifier.fit(X_train, y_train)
    y_train_pred = classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    rectangles = segmentation(X, y, verbose=verbose)

    X_train_arr = X_train.values
    X_test_arr = X_test.values
    y_test_arr = y_test.values
    
    test_only_segments = []

    for rect, label in rectangles:
        rect = np.array(rect)
        train_in_rect = np.all((X_train_arr >= rect[:, 0]) & (X_train_arr <= rect[:, 1]), axis=1)
        if np.any(train_in_rect):
            continue
        
        test_in_rect = np.all((X_test_arr >= rect[:, 0]) & (X_test_arr <= rect[:, 1]), axis=1)
        if not np.any(test_in_rect):
            continue
            
        test_only_segments.append((rect, label))
    
    correct_segments = []
    wrong_segments = []
    correct_samples = 0
    wrong_samples = 0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        
        for rect, seg_label in test_only_segments:
            lower_bounds = rect[:, 0]
            upper_bounds = rect[:, 1]
            mask = np.all((X_test_arr >= lower_bounds) & (X_test_arr <= upper_bounds), axis=1)

            segment_samples = X_test_arr[mask]
            num_samples = len(segment_samples)
            preds = classifier.predict(segment_samples)

            if np.all(preds == seg_label):
                correct_segments.append((rect, seg_label))
                correct_samples += num_samples
            else:
                wrong_segments.append((rect, seg_label))
                wrong_samples += num_samples
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': accuracy,
        'total_mini_clusters': len(rectangles),
        'correct_segments': len(correct_segments),
        'wrong_segments': len(wrong_segments),
        'correct_samples': correct_samples,
        'wrong_samples': wrong_samples,
        'total_test_only_segments': len(test_only_segments),
        'train_accuracy': train_accuracy,
        'test_accuracy': accuracy
    }

# function to select classifier
def get_classifier(classifier_name):
    if classifier_name == "Gaussian Naive Bayes":
        return GaussianNB()
    elif classifier_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    elif classifier_name == "SVM":
        return SVC(random_state=42)
    elif classifier_name == "Random Forest":
        return RandomForestClassifier(random_state=42)

# Initialize DataLoader
@st.cache_resource
def get_data_loader():
    return DataLoader()

# Streamlit App
st.set_page_config(page_title="Overfitting Analyzer", layout="wide")
st.title("🔍 Overfitting Analyzer")
st.markdown("---")

# Initialize data loader
try:
    data_loader = get_data_loader()
    available_datasets = data_loader.get_available_datasets()
except Exception as e:
    st.error(f"Error initializing DataLoader: {str(e)}")
    st.stop()

# Sidebar for inputs and configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Dataset selection method
    dataset_source = st.radio("📂 Dataset Source", ["Internal Datasets", "Upload File"])
    
    df = None
    
    if dataset_source == "Internal Datasets":
        # Select from internal datasets
        selected_dataset = st.selectbox("Select Dataset", available_datasets)
        
        if st.button("📥 Load Dataset", type="secondary"):
            try:
                with st.spinner(f"Loading {selected_dataset}..."):
                    df = data_loader.load_dataset(selected_dataset)
                    st.session_state['df'] = df
                    st.session_state['dataset_name'] = selected_dataset
                    st.success(f"✅ {selected_dataset} loaded!")
                    st.info(f"Shape: {df.shape}")
            except Exception as e:
                st.error(f"❌ Error loading dataset: {str(e)}")
    else:
        # File upload
        uploaded_file = st.file_uploader("Upload Dataset (CSV/Excel)", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                
                if 'class' not in df.columns:
                    st.error("❌ Dataset must have a 'class' column")
                    df = None
                else:
                    st.session_state['df'] = df
                    st.session_state['dataset_name'] = uploaded_file.name
                    st.success(f"✅ Dataset loaded!")
                    st.info(f"Shape: {df.shape}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.markdown("---")
    
    # show the configuration (test split size, classifier, number of runs) if dataset is loaded
    if 'df' in st.session_state:
        st.subheader("🎯 Analysis Parameters")
        
        # select split size
        test_size = st.selectbox("Test Split Size", [0.2, 0.3, 0.4, 0.5, 0.6], index=1)
        
        # Select Classifier
        classifier_name = st.selectbox("Classifier", 
                                       ["Gaussian Naive Bayes", "Logistic Regression", 
                                        "SVM", "Random Forest"])
        
        # Number of runs
        num_runs = st.number_input("Number of Runs", min_value=1, max_value=200, value=50)
        
        st.markdown("---")
        
        # Run button
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    else:
        run_analysis = False

# analysis sections
if 'df' in st.session_state:
    df = st.session_state['df']
    dataset_name = st.session_state.get('dataset_name', 'Unknown')
    
    # Dataset info section
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Dataset", dataset_name)
    with col2:
        st.metric("📏 Rows", df.shape[0])
    with col3:
        st.metric("📐 Features", df.shape[1] - 1)  # Excluding 'class' column
    
    # Show dataset preview
    with st.expander("📊 Dataset Preview & Info"):
        tab1, tab2, tab3 = st.tabs(["Preview", "Statistics", "Class Distribution"])
        
        with tab1:
            st.dataframe(df.head(20), use_container_width=True)
        
        with tab2:
            st.write(df.describe())
        
        with tab3:
            class_counts = df['class'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 4))
            class_counts.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Class Distribution')
            plt.tight_layout()
            st.pyplot(fig)
    
    if run_analysis:
        st.markdown("---")
        st.header("🔬 Running Analysis...")
        
        with st.spinner("Processing..."):
            # Run analysis
            total_mini_clusters = 0
    
            correct_segments = []
            wrong_segments = []
            correct_samples = []
            wrong_samples = []
            segment_ratios = []
            sample_ratios = []
            train_accuracies = []
            test_accuracies = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(num_runs):
                status_text.text(f"Run {i+1}/{num_runs}")
                classifier = get_classifier(classifier_name)
                
                try:
                    results = is_overfitting(df, classifier, test_size=test_size, 
                                            random_state=i, verbose=False)
                    if results['wrong_segments'] == 0:
                        continue
                    total_mini_clusters = results['total_mini_clusters']
                    train_accuracies.append(results['train_accuracy'])
                    test_accuracies.append(results['test_accuracy'])
                    correct_segments.append(results['correct_segments'])
                    wrong_segments.append(results['wrong_segments'])
                    correct_samples.append(results['correct_samples'])
                    wrong_samples.append(results['wrong_samples'])
                    # ratio calculations
                    segment_ratio = results['correct_segments'] / results['wrong_segments']
                    sample_ratio = results['correct_samples'] / results['wrong_samples']
                    segment_ratios.append(segment_ratio)
                    sample_ratios.append(sample_ratio)
                    
                except Exception as e:
                    st.warning(f"Run {i+1} failed: {str(e)}")
                    continue
                
                progress_bar.progress((i + 1) / num_runs)
            
            progress_bar.empty()
            status_text.empty()
            
            if len(segment_ratios) == 0:
                st.error("❌ No valid results. All runs had zero wrong segments.")
            else:
                # Calculate statistics
                avg_correct_segments = np.mean(correct_segments)
                std_correct_segments = np.std(correct_segments)
                avg_wrong_segments = np.mean(wrong_segments)
                std_wrong_segments = np.std(wrong_segments)
                avg_correct_samples = np.mean(correct_samples)
                std_correct_samples = np.std(correct_samples)
                avg_wrong_samples = np.mean(wrong_samples)
                std_wrong_samples = np.std(wrong_samples)
                avg_segment_ratio = np.mean(segment_ratios)
                avg_sample_ratio = np.mean(sample_ratios)
                avg_train_acc = np.mean(train_accuracies)
                avg_test_acc = np.mean(test_accuracies)
                std_train_acc = np.std(train_accuracies)
                std_test_acc = np.std(test_accuracies)
                
                
                overfitting_segment = "Not Overfitting" if avg_segment_ratio > 0.5 else "Overfitting"
                overfitting_sample = "Not Overfitting" if avg_sample_ratio > 0.5 else "Overfitting"
                
                # Display results
                st.markdown("---")
                st.header("📈 Results")

                # 1st rowmetrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Correct Segments", f"{avg_correct_segments:.4f}",
                              delta=f"±{std_correct_segments:.4f}")
                with col2:
                    st.metric("Avg Wrong Segments", f"{avg_wrong_segments:.4f}",
                              delta=f"±{std_wrong_segments:.4f}")
                with col3:
                    st.metric("Avg Correct Samples", f"{avg_correct_samples:.4f}",
                              delta=f"±{std_correct_samples:.4f}")
                with col4:
                    st.metric("Avg Wrong Samples", f"{avg_wrong_samples:.4f}",
                              delta=f"±{std_wrong_samples:.4f}")

                # 2nd row metrics
                col5, col6, col7, col8 = st.columns(4)
                with col5:
                    st.metric("Avg Train Accuracy", f"{avg_train_acc:.4f}", 
                             delta=f"±{std_train_acc:.4f}")
                with col6:
                    st.metric("Avg Test Accuracy", f"{avg_test_acc:.4f}", 
                             delta=f"±{std_test_acc:.4f}")
                with col7:
                    st.metric("Segment Ratio", f"{avg_segment_ratio:.4f}")
                with col8:
                    st.metric("Sample Ratio", f"{avg_sample_ratio:.4f}")
                
                st.markdown("### 🎯 Overfitting Detection")
                col1, col2 = st.columns(2)
                with col1:
                    if overfitting_segment == "Overfitting":
                        st.error(f"🔴 **Segment-wise:** {overfitting_segment}")
                    else:
                        st.success(f"🟢 **Segment-wise:** {overfitting_segment}")
                with col2:
                    if overfitting_sample == "Overfitting":
                        st.error(f"🔴 **Sample-wise:** {overfitting_sample}")
                    else:
                        st.success(f"🟢 **Sample-wise:** {overfitting_sample}")
                
                # Plot density
                st.markdown("---")
                st.header("📊 Density Plots")
                
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                
                # Plot 1: Segment Ratios
                sns.kdeplot(segment_ratios, fill=True, ax=axes[0], color='blue', alpha=0.6)
                axes[0].axvline(avg_segment_ratio, color='red', linestyle='--', linewidth=2, 
                               label=f'Mean: {avg_segment_ratio:.4f}')
                axes[0].axvline(0.5, color='green', linestyle='--', linewidth=2, label='Threshold: 0.5')
                axes[0].set_xlabel('Segment Ratio (Correct/Wrong)', fontsize=11)
                axes[0].set_ylabel('Density', fontsize=11)
                axes[0].set_title('Density Plot of Segment Ratios', fontsize=13, fontweight='bold')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Plot 2: Sample Ratios
                sns.kdeplot(sample_ratios, fill=True, ax=axes[1], color='orange', alpha=0.6)
                axes[1].axvline(avg_sample_ratio, color='red', linestyle='--', linewidth=2, 
                               label=f'Mean: {avg_sample_ratio:.4f}')
                axes[1].axvline(0.5, color='green', linestyle='--', linewidth=2, label='Threshold: 0.5')
                axes[1].set_xlabel('Sample Ratio (Correct/Wrong)', fontsize=11)
                axes[1].set_ylabel('Density', fontsize=11)
                axes[1].set_title('Density Plot of Sample Ratios', fontsize=13, fontweight='bold')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                
                
                # Download results
                st.markdown("---")
                st.header("💾 Download Results")
                
                results_df = pd.DataFrame({
                    'Run': range(1, len(segment_ratios) + 1),
                    'Segment_Ratio': segment_ratios,
                    'Sample_Ratio': sample_ratios,
                    'Train_Accuracy': train_accuracies,
                    'Test_Accuracy': test_accuracies
                })
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"overfitting_analysis_{dataset_name}_{classifier_name}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
                # Summary statistics
                with st.expander("📊 Detailed Statistics"):
                    summary_stats = {
                        'Metric': ['Valid Runs', 'Avg Train Acc', 'Std Train Acc', 'Avg Test Acc', 
                                  'Std Test Acc', 'Avg Segment Ratio', 'Avg Sample Ratio'],
                        'Value': [len(segment_ratios), f"{avg_train_acc:.4f}", f"{std_train_acc:.4f}",
                                 f"{avg_test_acc:.4f}", f"{std_test_acc:.4f}", 
                                 f"{avg_segment_ratio:.4f}", f"{avg_sample_ratio:.4f}"]
                    }
                    st.table(pd.DataFrame(summary_stats))
else:
    # Welcome screen
    st.info("👈 Please select or upload a dataset from the sidebar to begin analysis")
    
    st.markdown("""
    ### 📖 How to use:
    1. **Select a dataset** from the internal datasets or upload your own CSV/Excel file
    2. **Configure parameters** (test size, classifier, number of runs)
    3. **Run the analysis** and view the results
    4. **Download** the results for further analysis
    
    ### 📊 Available Internal Datasets:
    """)
    
    # Display available datasets in columns
    cols = st.columns(3)
    for idx, dataset in enumerate(available_datasets):
        with cols[idx % 3]:
            st.markdown(f"- {dataset}")