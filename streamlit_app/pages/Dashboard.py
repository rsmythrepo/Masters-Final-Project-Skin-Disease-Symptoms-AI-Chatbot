import streamlit as st
st.set_page_config(page_title="My App",layout='wide')
st.title("Dashboard")

margins_css = """
    <style>
.appview-container .main .block-container{{
        padding-top: {padding_top}rem;    }}
</style>
"""

st.markdown(margins_css, unsafe_allow_html=True)

st.markdown("### Infrastructure Diagram")
st.image('../streamlit_app/dashboard images/infra_diagram.png', caption='Infrastructure', use_column_width=True)
st.markdown("### Inference System Chatbot")
st.image('../streamlit_app/dashboard images/inference_chatbot.png', caption='Inference System Chatbot', use_column_width=True)
st.markdown("### Inference System Image Classifier")
st.image('../streamlit_app/dashboard images/Images Inference System.drawio.png', caption='Inference System Images', use_column_width=True)


st.markdown("### Model Performance")
col3, col4 = st.columns([1,1])
col3.markdown("#### Performance Comparisons")
col3.image('../streamlit_app/dashboard images/performance tables .png', caption='Performance Tables', use_column_width=True)
col4.markdown("#### Chosen Model")
col4.markdown("##### Xception Model")
col4.image('../streamlit_app/dashboard images/best model.png', caption='Best Model Performance', use_column_width=True)


