import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy.stats import gaussian_kde


def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Devuelve una lista con las columnas numericas del DataFrame.
    """
    return df.select_dtypes(include=["number"]).columns.tolist()


def plot_histogram(data: pd.Series, variable_name: str) -> None:
    """
    Muestra un histograma de la variable seleccionada.
    """
    fig, ax = plt.subplots()
    ax.hist(data.dropna(), bins=20)
    ax.set_title(f"Histograma de {variable_name}")
    ax.set_xlabel(variable_name)
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)


def plot_kde(data: pd.Series, variable_name: str) -> None:
    """
    Muestra una curva KDE de la variable seleccionada.
    """
    clean_data = data.dropna()

    if len(clean_data) < 2:
        st.warning("No hay suficientes datos para calcular la KDE.")
        return

    kde = gaussian_kde(clean_data)
    x_values = pd.Series(clean_data).sort_values()
    y_values = kde(x_values)

    fig, ax = plt.subplots()
    ax.plot(x_values, y_values)
    ax.set_title(f"KDE de {variable_name}")
    ax.set_xlabel(variable_name)
    ax.set_ylabel("Densidad")
    st.pyplot(fig)


def plot_boxplot(data: pd.Series, variable_name: str) -> None:
    """
    Muestra un boxplot de la variable seleccionada.
    """
    fig, ax = plt.subplots()
    ax.boxplot(data.dropna(), vert=True)
    ax.set_title(f"Boxplot de {variable_name}")
    ax.set_ylabel(variable_name)
    st.pyplot(fig)