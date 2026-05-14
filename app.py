import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)

from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score
)

# ==========================================
# TITLE
# ==========================================

st.title("Decision Tree ML Web App")
st.write("Upload your own dataset for Classification or Regression")

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read Dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ==========================================
    # SELECT TASK
    # ==========================================

    task = st.selectbox(
        "Select Task",
        ["Classification", "Regression"]
    )

    # ==========================================
    # TARGET COLUMN
    # ==========================================

    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    # ==========================================
    # FEATURES & TARGET
    # ==========================================

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ==========================================
    # HANDLE CATEGORICAL DATA
    # ==========================================

    X = pd.get_dummies(X)

    # ==========================================
    # SPLIT DATA
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ==========================================
    # CLASSIFICATION
    # ==========================================

    if task == "Classification":

        st.header("Decision Tree Classification")

        params = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [1, 2, 3, 4, 5, 6],
            'splitter': ['best', 'random'],
            'max_features': ['sqrt', 'log2', None]
        }

        classifier = DecisionTreeClassifier()

        grid = GridSearchCV(
            estimator=classifier,
            param_grid=params,
            cv=5,
            scoring='accuracy'
        )

        if st.button("Train Classification Model"):

            grid.fit(X_train, y_train)

            y_pred = grid.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            st.subheader("Best Parameters")
            st.write(grid.best_params_)

            st.subheader("Best Cross Validation Score")
            st.write(grid.best_score_)

            st.subheader("Accuracy")
            st.write(accuracy)

            st.subheader("Confusion Matrix")
            st.write(confusion_matrix(y_test, y_pred))

            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))

    # ==========================================
    # REGRESSION
    # ==========================================

    else:

        st.header("Decision Tree Regression")

        params = {
            'criterion': [
                'squared_error',
                'friedman_mse',
                'absolute_error'
            ],
            'splitter': ['best', 'random'],
            'max_depth': [1, 2, 3, 4, 5, 6],
            'max_features': ['sqrt', 'log2', None]
        }

        regressor = DecisionTreeRegressor()

        random_search = RandomizedSearchCV(
            estimator=regressor,
            param_distributions=params,
            n_iter=10,
            cv=5,
            scoring='r2',
            random_state=42
        )

        if st.button("Train Regression Model"):

            random_search.fit(X_train, y_train)

            y_pred = random_search.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)

            r2 = r2_score(y_test, y_pred)

            st.subheader("Best Parameters")
            st.write(random_search.best_params_)

            st.subheader("Best Cross Validation Score")
            st.write(random_search.best_score_)

            st.subheader("Mean Squared Error")
            st.write(mse)

            st.subheader("R2 Score")
            st.write(r2)

# ==========================================
# FOOTER
# ==========================================

st.write("Built using Streamlit and Scikit-learn")