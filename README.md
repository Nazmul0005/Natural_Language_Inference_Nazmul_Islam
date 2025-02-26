# Sentence Pair Classification Project

## Project Overview
This project focuses on building a machine learning model that classifies pairs of sentences into three categories:
- **Contradiction (0):** Sentences with opposite meanings.
- **Neutral (1):** Sentences that are related but do not imply one another.
- **Entailment (2):** One sentence logically follows from the other.
  
###  **Example of NLI**
  - **Premise:** "A man is playing a guitar on stage."

  - **Hypotheses and Their Labels:-**

          ✅ Entailment: "A musician is performing live."
          
          ❌ Contradiction: "No one is playing music."
          
          🔸 Neutral: "A crowd is cheering."


The goal is to perform thorough exploratory data analysis (EDA), implement robust text preprocessing, experiment with several model architectures (baseline machine learning, ANN, LSTM, and Transformer-based models), and evaluate their performance.

## Dataset Description
The dataset consists of:
- **train.csv:** Contains labeled sentence pairs with the following columns:
  - `id`: Unique identifier for each example.
  - `premise`: The first sentence.
  - `hypothesis`: The second sentence.
  - `label`: The class label (0: Contradiction, 1: Neutral, 2: Entailment)
- **test.csv:** Created by splitting the `train.csv` (e.g., 80% training and 20% testing) to simulate unseen data for evaluation.

## Model Implementation Details
The project involves the following steps:

1. **Exploratory Data Analysis (EDA):**
   - Load the dataset and perform data quality checks (missing values, duplicates).
   - Visualize label distribution, sentence lengths, and word frequency (using histograms and word clouds).

2. **Text Preprocessing:**
   - **Tokenization:** Splitting sentences into words.
   - **Normalization:** Converting text to lowercase, removing punctuation and stop words.
   - **Stemming/Lemmatization:** Reducing words to their root forms.
   - **Feature Extraction:** Using approaches such as TF-IDF vectors and word embeddings.

3. **Model Creation:**
   - **Baseline Models:** Implement traditional classifiers (e.g., Random Forest) with TF-IDF features.
   - **Neural Network:** Develop a simple ANN using Keras.
   - **Sequence Models:** Use LSTM networks to capture sequential information.
   - **Transformer Models:** Fine-tune pre-trained models like BERT for enhanced contextual understanding.

4. **Model Evaluation:**
   - Evaluate models using metrics such as accuracy, precision, recall, F1-score, and confusion matrices.
   - Visualize performance with training curves and ROC curves (if applicable).

5. **Model Tuning and Optimization:**
   - Conduct hyperparameter tuning using techniques like Grid Search and early stopping.
   - Experiment with optimizers, learning rates, batch sizes, and regularization techniques.

## Steps to Run the Code
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/sentence-pair-classification.git
   cd sentence-pair-classification
   ```
2. **Install Dependencies: Ensure you have Python installed, then run:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Data Preparation:**
   - Place the provided train.csv into the data/ directory.
   - Run the script in src/data_preprocessing.py to create the test.csv (if not already provided).

4. **Run Notebooks:**
   - Open notebooks/Data_Exploration.ipynb to view the EDA and visualization steps.
   - Open notebooks/Model_Training.ipynb to run the model training and evaluation pipelines.
6. **Review Performance:**
   - See the detailed performance evaluation in performance_report.pdf (or performance_report.md) for model metrics and observations.


## Model Evaluation Results
- Baseline (Random Forest): Achieved an accuracy of X% with detailed metrics such as precision, recall, and F1-score.
- Neural Network (ANN): Achieved an accuracy of Y% after tuning hyperparameters.
- LSTM Model: Achieved an accuracy of Z%, demonstrating effectiveness in capturing sequential dependencies.
- Transformer Model (BERT): Achieved the highest accuracy of W% due to superior contextual understanding.
- For full details, refer to the performance report (performance_report.pdf).

## Additional Observations or Notes
- Hyperparameter tuning was crucial in improving model performance. Techniques like early stopping helped prevent overfitting.
- The transformer-based model, although resource-intensive, showed significant improvements in contextual understanding.
- Future work may include exploring ensemble methods or additional pre-processing techniques to further enhance performance.
