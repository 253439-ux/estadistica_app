import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm


def plot_z_test_curve(z_stat: float, critical_value: float, tail_type: str) -> None:
    """
    Grafica la curva normal estandar, las regiones de rechazo
    y la posicion del estadistico Z.
    """
    x = np.linspace(-4, 4, 1000)
    y = norm.pdf(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, label="N(0,1)")

    if tail_type == "Izquierda":
        rejection_x = x[x <= critical_value]
        rejection_y = norm.pdf(rejection_x)
        ax.fill_between(rejection_x, rejection_y, alpha=0.3)

    elif tail_type == "Derecha":
        rejection_x = x[x >= critical_value]
        rejection_y = norm.pdf(rejection_x)
        ax.fill_between(rejection_x, rejection_y, alpha=0.3)

    elif tail_type == "Bilateral":
        left_x = x[x <= -critical_value]
        left_y = norm.pdf(left_x)
        ax.fill_between(left_x, left_y, alpha=0.3)

        right_x = x[x >= critical_value]
        right_y = norm.pdf(right_x)
        ax.fill_between(right_x, right_y, alpha=0.3)

        ax.axvline(-critical_value, linestyle="--", label=f"-Z critico = {-critical_value:.2f}")

    ax.axvline(critical_value, linestyle="--", label=f"Z critico = {critical_value:.2f}")
    ax.axvline(z_stat, linestyle="-", label=f"Z observado = {z_stat:.2f}")

    ax.set_title("Curva normal estandar y regiones de rechazo")
    ax.set_xlabel("Z")
    ax.set_ylabel("Densidad")
    ax.legend()

    st.pyplot(fig)