# app.py - Decision Tree Classifier Streamlit App

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# load dataset
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# train model
model = DecisionTreeClassifier(criterion="gini", max_depth=3)
model.fit(X, y)

st.title("Decision Tree Classifier - Iris Dataset")

st.write("Enter input values:")

sepal_length = st.number_input("Sepal Length (cm)")
sepal_width = st.number_input("Sepal Width (cm)")
petal_length = st.number_input("Petal Length (cm)")
petal_width = st.number_input("Petal Width (cm)")

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    species = iris.target_names[prediction][0]

    st.success(f"Predicted Class: {species}")