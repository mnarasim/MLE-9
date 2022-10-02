import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import matplotlib.pyplot as plt
import shap

# Add and resize an image to the top of the app
img_fuel = Image.open("../img/fuel_efficiency.png")
st.image(img_fuel, width=700)

st.markdown("<h1 style='text-align: center; color: black;'>Fuel Efficiency</h1>", unsafe_allow_html=True)

# Import train dataset to DataFrame
#train_df = pd.read_csv("../dat/train.csv.gz", compression="gzip")
#model_results_df = pd.read_csv("../dat/model_results.csv")



train_df = pd.read_csv('train.csv', index_col = 0)
results = pd.read_csv('results.csv', index_col = 0)
tpot_results= pd.read_csv('tpot.csv', index_col = 0)
shap1 = pd.read_csv('shap1.csv', index_col = 0)

shap2 = pd.read_csv('shap2.csv', index_col = 0)


# Create sidebar for user selection
with st.sidebar:
    # Add FB logo
    st.image("https://user-images.githubusercontent.com/37101144/161836199-fdb0219d-0361-4988-bf26-48b0fad160a3.png" )    

    # Available models for selection

    # YOUR CODE GOES HERE!
    models = ["DNN", "TPOT"]

    # Add model select boxes
    model1_select = st.selectbox(
        "Choose Model 1:",
        (models)
    )
    
    # Remove selected model 1 from model list
    # App refreshes with every selection change.
    models.remove(model1_select)
    
    model2_select = st.selectbox(
        "Choose Model 2:",
        (models)
    )

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
    
    

    # Columns for side-by-side model comparison
    col1, col2 = st.columns(2)

    # Build the confusion matrix for the first model.
    with col1:
        st.header(model1_select)

        st.dataframe(results.head())


    # Build confusion matrix for second model
    with col2:
        st.header(model2_select)

        
        st.dataframe(tpot_results.head())


with tab3: 
    # YOUR CODE GOES HERE!
        # Use columns to separate visualizations for models
        # Include plots for local and global explanability!
        
        # Summary plot 1 SHAP


    # Summary plot 1 SHAP

     
    st.header(model1_select)
    st.subheader('Summary Plot -Simple NN')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap1.to_numpy(), train_df.columns)
    st.pyplot(fig)
    
    st.header(model2_select)
    st.subheader('Summary Plot - Deep NN')
    fig, ax = plt.subplots(nrows=1, ncols=1)
    shap.summary_plot(shap2.to_numpy(), train_df.columns)
    st.pyplot(fig)

    
