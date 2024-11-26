# FIFA Player Overall Rating Prediction

This project aims to predict the overall rating of FIFA players based on their individual attributes using machine learning techniques. By leveraging players' characteristics and personal information, the model estimates their overall performance, which can be useful for game analysis, scouting, and strategic planning.

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Overview](#model-overview)
- [Results](#results)
- [License](#license)

## Dataset

The project uses two datasets:

- **`caracteristicas_jogadores.csv`**: Contains players' attributes over different dates.
- **`nomes_jogadores.csv`**: Contains players' personal information such as name, birthday, height, and weight.

## Project Structure

- **`main.py`**: The main script containing data preprocessing, model training, evaluation, and saving.
- **`caracteristicas_jogadores.csv`**: Dataset with players' attributes.
- **`nomes_jogadores.csv`**: Dataset with players' personal information.
- **`pipeline_rf_model.pkl`**: The saved Random Forest model for future predictions.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- joblib

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd your-repo-name
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not provided, install packages individually:*

   ```bash
   pip install pandas numpy scikit-learn xgboost joblib
   ```

## Usage

1. **Ensure the datasets are in the project directory:**

   - `caracteristicas_jogadores.csv`
   - `nomes_jogadores.csv`

2. **Run the script main.ipynb:**

   The script will:

   - Preprocess the data.
   - Train multiple machine learning models.
   - Evaluate each model's performance.
   - Save the best-performing model (`pipeline_rf_model.pkl`).

## Model Overview

### Data Preprocessing

- **Date and Birthday Processing:**

  - Extracted the year from the `date` and `birthday` columns.
  - Calculated players' `age` by subtracting `birthday` from `date`.

- **Data Cleaning:**

  - Merged the two datasets on `player_fifa_api_id`.
  - Removed rows with missing values.
  - Filtered out rows where `defensive_work_rate` is not `low`, `medium`, or `high`.

- **Feature Selection:**

  Selected relevant features for modeling, including:

  - Technical attributes (e.g., `crossing`, `finishing`, `dribbling`).
  - Physical attributes (e.g., `height`, `weight`, `age`).
  - Work rates (`attacking_work_rate`, `defensive_work_rate`).

- **Encoding Categorical Variables:**

  - Converted `attacking_work_rate` and `defensive_work_rate` to categorical types.
  - Applied One-Hot Encoding to these categorical features.

- **Feature Scaling:**

  - Scaled numerical features using `StandardScaler`.

### Model Training and Evaluation

- **Data Splitting:**

  - Split the data into training (60%), validation (20%), and testing (20%) sets.

- **Models Used:**

  1. **Random Forest Regressor**
  2. **Random Forest Regressor** with hyperparameter tuning (`n_estimators=200`, `max_depth=15`)
  3. **Gradient Boosting Regressor**
  4. **XGBoost Regressor**
  5. **XGBoost Regressor** with hyperparameter tuning (`n_estimators=150`, `learning_rate=0.05`)
  6. **Stacking Regressor** combining the above models

- **Performance Metrics:**

  - Weighted Mean Absolute Percentage Error (WMAPE)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)

- **Model Selection:**

  - Evaluated each model on the validation set.
  - Selected the **Random Forest Regressor** as the final model based on performance metrics.

### Saving the Model

- Trained the final model on the combined training and validation sets.
- Evaluated on the test set to ensure generalization.
- Saved the trained model using `joblib` for future predictions.

## Results

*Note: Specific numerical results are not provided as they depend on the actual execution of the script and datasets.*

- The **Random Forest Regressor** demonstrated superior performance with lower error metrics compared to other models.
- The model effectively predicts the overall rating of FIFA players based on their attributes and personal information.