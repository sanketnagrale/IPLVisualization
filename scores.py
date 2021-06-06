import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


def scores_function():
    pr = st.sidebar.slider("Select Hours to Predict Percentage", 1.00, 15.00, 2.5)
    st.title("Score Prediction")
    st.write("Data")
    scores = pd.read_csv('scores.csv')
    st.write(scores)
    regr = linear_model.LinearRegression()
    x = scores['Hours'].to_numpy().reshape(-1, 1)
    x_train = x[:20]
    y = scores['Scores'].to_numpy().reshape(-1, 1)
    y_train = y[:20]
    x_test = x[20:]
    y_test = y[20:]
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_test)
    st.write('Coefficient: \n', regr.coef_[0][0])
    st.write('Mean squared error: %.2f'
             % mean_squared_error(y_test, y_pred))
    st.write('Coefficient of determination: %.2f'
             % r2_score(y_test, y_pred))
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(x_train, y_train, color='orange', label="Training Data")
    plt.scatter(x_test, y_test,  color='black', label="Test Data")
    plt.plot(x_test, y_pred, color='blue', linestyle='--', label="Predicted")
    plt.legend()
    plt.xticks()
    plt.yticks()
    st.sidebar.write("Predicted Percentage is: ", str(regr.predict([[pr]]))[2:-8])
    st.pyplot(fig)
