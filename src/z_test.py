import math
import pandas as pd
from scipy.stats import norm


def run_z_test(
    data: pd.Series,
    mu_0: float,
    sigma: float,
    alpha: float,
    tail_type: str
) -> dict:
    """
    Ejecuta una prueba Z para una muestra.
    Supone sigma conocida y n >= 30.
    """
    clean_data = data.dropna()
    n = len(clean_data)

    if n < 30:
        raise ValueError("La prueba Z requiere n >= 30.")

    if sigma <= 0:
        raise ValueError("La desviacion estandar poblacional debe ser mayor que 0.")

    sample_mean = clean_data.mean()
    standard_error = sigma / math.sqrt(n)
    z_stat = (sample_mean - mu_0) / standard_error

    if tail_type == "Izquierda":
        p_value = norm.cdf(z_stat)
        critical_value = norm.ppf(alpha)
        reject = z_stat < critical_value

    elif tail_type == "Derecha":
        p_value = 1 - norm.cdf(z_stat)
        critical_value = norm.ppf(1 - alpha)
        reject = z_stat > critical_value

    elif tail_type == "Bilateral":
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
        critical_value = norm.ppf(1 - alpha / 2)
        reject = abs(z_stat) > critical_value

    else:
        raise ValueError("Tipo de cola no valido.")

    decision = "Rechazar H0" if reject else "No rechazar H0"

    return {
        "n": n,
        "sample_mean": sample_mean,
        "standard_error": standard_error,
        "z_stat": z_stat,
        "p_value": p_value,
        "critical_value": critical_value,
        "decision": decision
    }