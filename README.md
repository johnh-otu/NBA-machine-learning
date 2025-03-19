# NBA Lineup Prediction

This project is designed to predict the **fifth player** for a home team in an NBA game using machine learning techniques. The model leverages historical NBA matchup data to predict the missing lineup player based on past performance metrics.

---

## Project Overview

This project was created for the **Machine Learning & Data Mining** course (SOFE 4620U) at Ontario Tech University. The primary goal is to predict the fifth starting player for an NBA team using past game data.

**Key Features:**

- Utilizes **Random Forest Classifier** for optimal performance.
- Implements **feature engineering** to assign player scores based on stats such as points, assists, and turnovers.
- Includes **data cleaning**, **encoding**, and a robust evaluation pipeline.
- Achieves **30.9% accuracy** with a Mean Squared Error (MSE) of **4260.106**.

---

## Project Structure

├── data/
│ ├── matchups-2007.csv
│ ├── matchups-2008.csv
│ ├── ...
│ ├── NBA_test.csv
│ └── NBA_test_labels.csv
│
├── models/
│ ├── clf0.joblib.pkl
│ ├── clf1.joblib.pkl
│ ├── clf2.joblib.pkl
│ ├── clf3.joblib.pkl
│ └── clf4.joblib.pkl
│
├── NBAAttempt2.ipynb
├── PlayerPreprocessing.ipynb
├── player_data.csv
├── combined_player_data.csv
└── README.md

---

## Installation

To run this project, you will need the following dependencies:

### **Required Libraries**

- Python 3.11 or higher
- Pandas
- NumPy
- Scikit-learn
- Joblib

### **Install Dependencies**

Run the following command to install the required packages:

```bash
pip install pandas numpy scikit-learn joblib
```

# Instructions to Run the Project

Follow these steps to run the NBA Lineup Prediction model successfully:

---

## Step 1: Clone the Repository

Clone the project from the GitHub repository:

```bash
git clone https://github.com/johnh-otu/NBA-machine-learning.git
cd NBA-machine-learning
```

## Step 2: Prepare the Dataset

- Place all the matchups-\*.csv files inside the /data folder.
- Ensure NBA_test.csv and NBA_test_labels.csv are also present.

## Step 3: Data Preprocessing

Run the PlayerPreprocessing.ipynb notebook to:

- Combine all matchup data.
- Calculate player scores.
- Save the processed player data as combined_player_data.csv.
-

## Step 4: Model Training

Run the `NBAAttempt2.ipynb` notebook to:

- Train five separate Random Forest models for each `home_0` to `home_4` position.
- Save the trained models as `.joblib.pkl` files in the `/models` folder.

---

## Step 5: Model Evaluation

In the same notebook:

- Load the test data (`NBA_test.csv`).
- Apply the trained models to predict the missing fifth home player.
- Evaluate performance using accuracy and mean squared error.

---

## Step 6: Results

The predicted results are automatically saved to a CSV file:

```bash
submission.csv
```

## How the Model Works

- The model predicts the missing fifth home player based on encoded features like team composition, starting minutes, and game outcomes.
- Each player's score is calculated using this formula:

## Performance

| **Metric**             | **Value** |
| ---------------------- | --------- |
| **Accuracy**           | 30.9%     |
| **Mean Squared Error** | 4260.106  |

---

## Known Issues

- **Low Accuracy**: While the model achieves 30.9% accuracy, exploring more complex models like **XGBoost** or **LightGBM** may improve results.
- **Handling Unknown Players**: Unseen players are treated with placeholder values. Improving the `handle_unknown_value` logic may improve predictions.
- **Model Overfitting**: Consider tuning hyperparameters further or adding more feature engineering steps to reduce overfitting.
