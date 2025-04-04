# xai-mimic-workspace

This project aims to explore Explainable AI (XAI) techniques using the MIMIC dataset and later transition to federated learning models. The project is structured to maintain separation of concerns, with dedicated directories for data, models, notebooks, and scripts.

## Project Structure

- **data/**: Contains the MIMIC dataset used for experimentation.
  - `mimic-dataset.csv`: The dataset file.

- **models/**: Contains model definitions.
  - `xai_model.py`: Defines the `XAIModel` class for training and prediction.
  - `federated_model.py`: Defines the `FederatedModel` class for federated learning.

- **notebooks/**: Contains Jupyter notebooks for analysis.
  - `exploratory_analysis.ipynb`: Notebook for exploratory data analysis.

- **scripts/**: Contains scripts for data processing and model training.
  - `preprocess_data.py`: Functions for data preprocessing.
  - `train_xai_model.py`: Script to train the XAI model.
  - `train_federated_model.py`: Script to train the federated learning model.

- **requirements.txt**: Lists the required Python dependencies for the project.

## Methodology

This project implements a comprehensive methodology for exploring Explainable AI (XAI) in a federated learning setting. The methodology consists of the following steps:

### 1. Data Preprocessing
- Load and clean the MIMIC dataset
- Handle missing values and outliers
- Normalize/standardize features
- Split data into training and testing sets
- Create federated data partitions for different clients

### 2. Model Architecture
- Implement a neural network model with configurable layers
- Support for both centralized and federated training
- Integration of XAI components (LIME and SHAP)
- Privacy-preserving mechanisms

### 3. XAI Implementation
- **LIME (Local Interpretable Model-agnostic Explanations)**
  - Generate local explanations for individual predictions
  - Configure sample size and feature selection
  - Aggregate explanations across clients
  - Apply privacy mechanisms to explanations

- **SHAP (SHapley Additive exPlanations)**
  - Compute feature importance using Shapley values
  - Handle both global and local explanations
  - Support for different background datasets
  - Privacy-preserving aggregation

### 4. Privacy Mechanisms
- Implement differential privacy using the Laplace mechanism
- Privacy parameters:
  - `epsilon`: Privacy budget (controls noise level)
  - `delta`: Failure probability
  - `clip_values`: Value range constraints
  - `noise_scale`: Scale of added noise
- Secure aggregation for federated learning

### 5. Experiment Design
The project includes several experiment configurations:

1. **Baseline**
   - Both LIME and SHAP enabled
   - No privacy mechanisms
   - Standard sampling parameters

2. **LIME-only**
   - Only LIME explanations
   - No privacy mechanisms
   - Increased LIME samples (2000)

3. **SHAP-only**
   - Only SHAP explanations
   - No privacy mechanisms
   - Increased SHAP samples (200)

4. **High Privacy LIME**
   - Only LIME explanations
   - Strict privacy settings (epsilon = 0.1)
   - Value clipping enabled
   - Secure aggregation

5. **High Privacy SHAP**
   - Only SHAP explanations
   - Strict privacy settings (epsilon = 0.1)
   - Value clipping enabled
   - Secure aggregation

### 6. Evaluation Metrics
- Model performance metrics:
  - Accuracy
  - AUC-ROC
  - F1-score
- Explanation quality metrics:
  - Feature importance consistency
  - Explanation stability
  - Privacy-utility tradeoff
- Federated learning metrics:
  - Client contribution
  - Communication efficiency
  - Convergence rate

### 7. Visualization and Analysis
- Training progress plots
- Feature importance visualizations
- Privacy impact analysis
- Client contribution analysis
- Explanation comparison across configurations

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd xai-mimic-workspace
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines

- Use the `exploratory_analysis.ipynb` notebook for initial data exploration.
- Run `preprocess_data.py` to prepare the dataset for modeling.
- Train the XAI model using `train_xai_model.py`.
- For federated learning, use `train_federated_model.py` after the initial XAI model training.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.