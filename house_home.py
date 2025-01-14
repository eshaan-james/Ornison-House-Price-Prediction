import streamlit as st

def app(house_df):
    with st.expander('View Database'):
        st.dataframe(house_df)
    st.subheader('Columns description')
    col_name, col_dtype, col_display = st.columns(3)
    if col_name.checkbox('Show all column names'):
        st.table(house_df.columns)
    if col_dtype.checkbox('View column datatype'):
        st.write(list(house_df.dtypes))
    if col_display.checkbox('View Column data'):
        col = st.selectbox('Select columns', house_df.columns)
        st.table(house_df[col])