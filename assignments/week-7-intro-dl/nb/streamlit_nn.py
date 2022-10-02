#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 21:27:13 2022

@author: mani
"""

import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import matplotlib.pyplot as plt
import shap


st.markdown("<h1 style='text-align: center; color: black;'>Deep Learning Model Results</h1>", unsafe_allow_html=True)


train_df = pd.read_csv('train.csv')
results = pd.read_csv('results.csv')

shap1 = pd.read_csv('shap1.csv')

shap2 = pd.read_csv('shap2.csv')



# Create tabs for separation of tasks
tab1, tab2, tab3 = st.tabs(["🗃 Data", "🔎 Model Results", "🤓 Model Explainability"])

with tab1:    
    # Data Section Header
    st.header("Raw Data")

    # Display first 100 samples of the dateframe
    st.dataframe(train_df.head(100))

    st.header("Correlations")

    # Heatmap
    corr = train_df.corr()
    fig = px.imshow(corr)
    st.write(fig)

with tab2:    
    st.dataframe(results.head())

with tab3:
    # Summary plot 1 SHAP
    st.subheader('Summary Plot -Simple NN')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap1.to_numpy(), train_df.columns)
    st.pyplot(fig)
    
    # Summary plot 1 SHAP
    st.subheader('Summary Plot - Deep NN')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap2.to_numpy(), train_df.columns)
    st.pyplot(fig)