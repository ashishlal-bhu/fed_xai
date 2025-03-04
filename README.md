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