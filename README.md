# Titanic Survival Prediction 🚢 (Experimental)
- This is an experimental project — this repo explores an approach to predicting Titanic survival that intentionally includes a target. For the "proper"/clean version of this model, see model.py

# Handled missing value🤹🏻
- Age → filled with mean
- Embarked → filled with mode
- Extracted deck from Cabin, and filled missing values with "U" (unknown)

# Predicted deck
- Used Random Forest model to predict deck from the other features
- Used the predicted deck values to update the deck column across the dataset (both missing and non-missing rows)

# Predicted survival ⛏️
- Used the predicted deck as a feature, and trained a Logistic Regression model to predict Survived
- Compared model performance with and without the deck feature


# Data Leakage 💦 (Important)

This version contains target leakage issue :

- When predicting deck, the Survived column was attached the features (it was never explicitly dropped)
- This means the deck-prediction model learned patterns from Survived as well

- That predicted deck — which now indirectly encodes information about Survived — was then used as a feature to predict Survived itself

# Tech Stack 🖥️
- Python
- pandas
- scikit-learn — RandomForestClassifier, LogisticRegression, ColumnTransformer, OneHotEncoder, StandardScaler, train_test_split
- matplotlib — for confusion matrix visualization

