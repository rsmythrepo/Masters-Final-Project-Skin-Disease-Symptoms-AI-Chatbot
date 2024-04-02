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

col1, col2 = st.columns([1,1])

col1.markdown("### Exploratory Data Analysis")
col11, col12 = col1.columns(2)
col11.markdown("##### DDI")
col11.image('../streamlit_app/dashboard images/DDI skintone.png', caption='Figure 1: Model Performance Tables', use_column_width=True)
col11.image('../streamlit_app/dashboard images/DDI mal.png', caption='Figure 1: Model Performance Tables', use_column_width=True)

col12.markdown("##### Fitzpatrick17k")
col12.image('../streamlit_app/dashboard images/combine skintone.png', caption='Figure 1: Model Performance Tables', use_column_width=True)
col12.image('../streamlit_app/dashboard images/combined mal.png', caption='Figure 1: Model Performance Tables', use_column_width=True)
col1.markdown("##### Combined Data")
col1.image('../streamlit_app/dashboard images/combine skintone.png', caption='Figure 1: Model Performance Tables', use_column_width=True)
#col1.markdown("##### Final Model")



col2.markdown("### Infrastructure Diagram")
col2.markdown("### Inference System Chatbot")
col2.image('../streamlit_app/dashboard images/infr_sys2.png', caption='Figure 1: Model Performance Tables', use_column_width=True)
col2.markdown("### Inference System Images")
col2.image('../streamlit_app/dashboard images/infr_sys.png', caption='Figure 1: Model Performance Tables', use_column_width=True)


st.markdown("### Model Performance")
col3, col4 = st.columns([1,1])
col3.markdown("#### Performance Comparisons")
col3.image('../streamlit_app/dashboard images/model testing results.png', caption='Figure 1: Model Performance Tables', use_column_width=True)
col4.markdown("#### Chosen Model")
col4.markdown("##### Xception Model")
col4.image('../streamlit_app/dashboard images/best model.png', caption='Figure 1: Model Performance Tables', use_column_width=True)


