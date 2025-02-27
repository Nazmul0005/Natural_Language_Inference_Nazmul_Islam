# Sentence Contradiction Classification

## Project Overview
This project focuses on building a machine learning model that classifies pairs of sentences into three categories:

- *Contradiction (0)*: Sentences with opposite meanings.
- *Neutral (1)*: Sentences that are related but do not imply one another.
- *Entailment (2)*: One sentence logically follows from the other.

The goal is to perform thorough exploratory data analysis (EDA), implement robust text preprocessing, experiment with several model architectures (baseline machine learning, ANN, LSTM, and Transformer-based models), and evaluate their performance.

## Example of NLI
*Premise*: "A man is playing a guitar on stage."

*Hypotheses and Their Labels*:
- ✅ *Entailment*: "A musician is performing live."
- ❌ *Contradiction*: "No one is playing music."
- 🔸 *Neutral*: "A crowd is cheering."

## Dataset Description
The dataset consists of:

- *train.csv*: Contains labeled sentence pairs with the following columns:
  - id: Unique identifier for each example.
  - premise: The first sentence.
  - hypothesis: The second sentence.
  - label: The class label (0: Contradiction, 1: Neutral, 2: Entailment)
- *evaluation data*: Created by splitting the train.csv (e.g., 80% training and 20% testing) to simulate unseen data for evaluation.

## Model Implementation Details
The project involves the following steps:

### Exploratory Data Analysis (EDA)
- Load the dataset and perform data quality checks (missing values, duplicates).
- Visualize label distribution, sentence lengths, and word frequency (using histograms and word clouds).

### Text Preprocessing
- *Tokenization*: Splitting sentences into words.
- *Normalization*: Converting text to lowercase, removing punctuation and stop words.
- *Stemming*: Reducing words to their root forms.
- *Feature Extraction*: Using approaches such as TF-IDF vectors and word embeddings.

### Model Creation
- *Baseline Models*: Implement traditional classifiers (Random Forest, Decision trees, XGB) with TF-IDF features.
- *Neural Network*: Develop a simple ANN using Keras.
- *Sequence Models*: Use LSTM networks to capture sequential information.
- *Transformer Models*: Fine-tune pre-trained models like XLM-R for enhanced contextual understanding.

### Model Evaluation
- Evaluated models using metrics such as accuracy, precision, recall, F1-score, and confusion matrices.

### Model Tuning and Optimization
- Conducted hyperparameter tuning using techniques like Grid Search and early stopping.
- Experimented with optimizers, learning rates, batch sizes, and regularization techniques.

## Project Structure

sentence-pair-classification/

├── data/  

│   ├── train.csv

├── notebooks/

│   ├── Data_Exploration.ipynb

├── README.md

├── requirements.txt

└── performance_report.pdf


## Steps to Run the Code

### 1. Clone the Repository
bash
git clone https://github.com/yourusername/sentence-pair-classification.git
cd sentence-pair-classification


### 2. Install Dependencies
Ensure you have Python installed, then run:
bash
pip install -r requirements.txt


### 3. Data Preparation
- Place the provided train.csv into the data/ directory.
- Run the script in src/data_preprocessing.py to create the test.csv (if not already provided).

### 4. Run Notebooks
- Open notebooks/Data_Exploration.ipynb to view the EDA and visualization as well as the model training and evaluation pipeline steps.

### 5. Review Performance
- See the detailed performance evaluation in performance_report.pdf for model metrics and observations.

## Model Evaluation Results
- *Baseline (Random Forest, Decision trees, XGB)*: Achieved an accuracy of 44%, 40% and 43% with detailed metrics such as precision, recall, and F1-score.
- *Neural Network (ANN)*: Achieved an accuracy of 49% after tuning hyperparameters.
- *LSTM Model*: Achieved an accuracy of 35%, demonstrating effectiveness in capturing sequential dependencies.
- *Transformer Model (XLM-R)*: Achieved the highest accuracy of 98% due to superior contextual understanding.

For full details, refer to the performance report (performance_report.pdf). As we can see XLM-R giving the best performance among all.

## Additional Observations or Notes
- Hyperparameter tuning was crucial in improving model performance. Techniques like early stopping helped prevent overfitting.
- The transformer-based model, although resource-intensive, showed significant improvements in contextual understanding.
- Future work may include exploring ensemble methods or additional pre-processing techniques to further enhance performance.

## Dependencies
- Python 3.x
- Pandas, NumPy
- Scikit-learn
- TensorFlow/Keras
- Transformers (Hugging Face)
- NLTK/SpaCy
- Matplotlib, Seaborn

## Contributors
- [Nazmul Islam]
