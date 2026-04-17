import pandas as pd
import streamlit as st


def load_csv(uploaded_file) -> pd.DataFrame:
    """
    Lee un archivo CSV subido desde Streamlit y devuelve un DataFrame.
    Si ocurre un error, muestra un mensaje en pantalla y devuelve un DataFrame vacio.
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        return pd.DataFrame()
