# Titanic Survival Prediction

## Overview
This project aims to build a machine learning model to predict survival outcomes of Titanic passengers using features such as age, gender, passenger class, fare, and more.

## Dataset
- **train.csv**: Contains passenger data along with survival outcomes (ground truth).
- **test.csv**: Contains passenger data without survival outcomes; used for final predictions.
- **gender_submission.csv**: A sample submission assuming all females survive, provided for reference.

## Approach
- Data cleaning and preprocessing (handling missing values, encoding categorical features).
- Feature engineering to improve model performance.
- Model training using Random Forest classifier.
- Model evaluation with accuracy on a validation set.
- Feature importance analysis.
- Generating a submission file compatible with the competition.

## How to Run
1. Ensure Python 3 and required libraries are installed (see `requirements.txt`).
2. Place the `train.csv` and `test.csv` files in the project folder.
3. Run the model script:

   ```bash
   python titanic_model.py

The script outputs:

Model accuracy on validation data.

A submission CSV file (submission.csv) with predictions for the test set.

Feature importance plot saved as an image.

**Results**
Validation Accuracy: ~82.68%

Public leaderboard score: ~0.73923 (your score may vary depending on model improvements)
