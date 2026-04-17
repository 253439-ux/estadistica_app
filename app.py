import pandas as pd
import streamlit as st

from src.data_loader import load_csv
from src.synthetic_data import generate_synthetic_data
from src.visualization import get_numeric_columns, plot_histogram, plot_kde, plot_boxplot
from src.z_test import run_z_test
from src.z_plot import plot_z_test_curve
from src.gemini_helper import ask_gemini

# --------------------------------------------------
# Configuracion inicial
# --------------------------------------------------

st.set_page_config(
    page_title="App Estadistica Interactiva",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# Estado inicial
# --------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

if "data_source" not in st.session_state:
    st.session_state.data_source = None

if "z_test_results" not in st.session_state:
    st.session_state.z_test_results = None

# --------------------------------------------------
# Titulo principal
# --------------------------------------------------
st.title("📊 Aplicacion Web de Estadistica")
st.markdown("Modulo 1: Carga de datos y estructura base en Streamlit.")

# --------------------------------------------------
# Sidebar - carga de datos
# --------------------------------------------------
st.sidebar.header("Modulo 1 · Carga de datos")

data_option = st.sidebar.radio(
    "Selecciona una fuente de datos:",
    options=["Subir CSV", "Generar datos sinteticos"]
)

# --------------------------------------------------
# Opcion 1: Subir CSV
# --------------------------------------------------
if data_option == "Subir CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Sube un archivo CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = load_csv(uploaded_file)

        if not df.empty:
            st.session_state.df = df
            st.session_state.data_source = f"CSV: {uploaded_file.name}"
            st.success("Archivo cargado correctamente.")

# --------------------------------------------------
# Opcion 2: Generar datos sinteticos
# --------------------------------------------------
elif data_option == "Generar datos sinteticos":
    n = st.sidebar.number_input(
        "Tamano de muestra",
        min_value=30,
        value=100,
        step=10
    )

    mean = st.sidebar.number_input(
        "Media",
        value=50.0,
        step=1.0
    )

    std = st.sidebar.number_input(
        "Desviacion estandar",
        min_value=0.1,
        value=10.0,
        step=0.5
    )

    seed = st.sidebar.number_input(
        "Semilla",
        min_value=0,
        value=42,
        step=1
    )

    if st.sidebar.button("Generar datos"):
        df = generate_synthetic_data(
            n=int(n),
            mean=float(mean),
            std=float(std),
            seed=int(seed)
        )
        st.session_state.df = df
        st.session_state.data_source = "Datos sinteticos"
        st.success("Datos sinteticos generados correctamente.")

# --------------------------------------------------
# Vista principal
# --------------------------------------------------
st.subheader("Resumen de datos cargados")

if not st.session_state.df.empty:
    st.write(f"**Fuente:** {st.session_state.data_source}")
    st.write(
        f"**Dimensiones:** "
        f"{st.session_state.df.shape[0]} filas x {st.session_state.df.shape[1]} columnas"
    )

    st.dataframe(st.session_state.df.head())

    st.subheader("Tipos de variables")

    dtypes_df = pd.DataFrame({
        "columna": st.session_state.df.columns,
        "tipo": st.session_state.df.dtypes.astype(str).values
    })

    st.dataframe(dtypes_df)

else:
    st.info(
        "Aun no has cargado datos. "
        "Usa la barra lateral para subir un CSV o generar datos sinteticos."
    )
    # --------------------------------------------------
# Modulo 2 - Visualizacion
# --------------------------------------------------
if not st.session_state.df.empty:
    st.subheader("Visualizacion de variables")

    numeric_columns = get_numeric_columns(st.session_state.df)

    if numeric_columns:
        selected_variable = st.selectbox(
            "Selecciona una variable numerica",
            options=numeric_columns
        )

        selected_data = st.session_state.df[selected_variable]

        st.markdown("### Histograma")
        plot_histogram(selected_data, selected_variable)

        st.markdown("### KDE")
        plot_kde(selected_data, selected_variable)

        st.markdown("### Boxplot")
        plot_boxplot(selected_data, selected_variable)
    else:
        st.warning("El dataset no contiene columnas numericas para visualizar.")
        # --------------------------------------------------
# Modulo 3 - Prueba Z
# --------------------------------------------------
if not st.session_state.df.empty:
    st.subheader("Prueba de hipotesis - Prueba Z")

    numeric_columns = get_numeric_columns(st.session_state.df)

    if numeric_columns:
        z_variable = st.selectbox(
            "Selecciona la variable para la prueba Z",
            options=numeric_columns,
            key="z_variable"
        )

        st.markdown("### Definicion de hipotesis")
        mu_0 = st.number_input("Valor de la media bajo H0 (mu_0)", value=50.0, step=1.0)
        sigma = st.number_input("Desviacion estandar poblacional conocida (sigma)", min_value=0.1, value=10.0, step=0.5)

        tail_type = st.radio(
            "Tipo de prueba",
            options=["Izquierda", "Derecha", "Bilateral"]
        )

        alpha = st.selectbox(
            "Nivel de significancia (alpha)",
            options=[0.10, 0.05, 0.01],
            index=1
        )

        if tail_type == "Izquierda":
            st.latex(r"H_0: \mu \geq \mu_0")
            st.latex(r"H_1: \mu < \mu_0")
        elif tail_type == "Derecha":
            st.latex(r"H_0: \mu \leq \mu_0")
            st.latex(r"H_1: \mu > \mu_0")
        else:
            st.latex(r"H_0: \mu = \mu_0")
            st.latex(r"H_1: \mu \neq \mu_0")

        if st.button("Ejecutar prueba Z"):
            try:
                results = run_z_test(
                    data=st.session_state.df[z_variable],
                    mu_0=float(mu_0),
                    sigma=float(sigma),
                    alpha=float(alpha),
                    tail_type=tail_type
                )
                st.session_state.z_test_results = {
                   "variable_name": z_variable,
                   "n": results["n"],
                   "sample_mean": results["sample_mean"],
                   "mu_0": float(mu_0),
                   "sigma": float(sigma),
                   "tail_type": tail_type,
                   "alpha": float(alpha),
                   "z_stat": results["z_stat"],
                   "p_value": results["p_value"],
                   "critical_value": results["critical_value"],
                   "decision": results["decision"]
                   }

                st.markdown("### Resultados estadisticos")
                st.write(f"**Tamano de muestra (n):** {results['n']}")
                st.write(f"**Media muestral:** {results['sample_mean']:.4f}")
                st.write(f"**Error estandar:** {results['standard_error']:.4f}")
                st.write(f"**Estadistico Z:** {results['z_stat']:.4f}")
                st.write(f"**p-value:** {results['p_value']:.6f}")
                st.write(f"**Valor critico:** {results['critical_value']:.4f}")
                st.write(f"**Decision:** {results['decision']}")

                st.markdown("### Visualizacion de la prueba Z")
                plot_z_test_curve(
                z_stat=results["z_stat"],
                critical_value=results["critical_value"],
                tail_type=tail_type
                )

            except ValueError as e:
                st.error(str(e))
    else:
        st.warning("No hay columnas numericas disponibles para la prueba Z.")

# --------------------------------------------------
# Modulo 4 - Interpretacion con Gemini
# --------------------------------------------------
st.subheader("Asistente estadistico con Gemini")

gemini_api_key = st.text_input(
    "Ingresa tu API Key de Gemini",
    type="password"
)

if st.session_state.z_test_results is not None:
    if st.button("Interpretar resultados con Gemini"):
        try:
            interpretation = ask_gemini(
                summary=st.session_state.z_test_results,
                api_key=gemini_api_key if gemini_api_key else None
            )

            st.markdown("### Interpretacion generada por Gemini")
            st.write(interpretation)

        except Exception as e:
            st.error(f"Error al consultar Gemini: {e}")
else:
    st.info("Primero ejecuta la prueba Z para generar un resumen interpretable por Gemini.")