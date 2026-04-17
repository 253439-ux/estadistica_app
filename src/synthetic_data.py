import numpy as np
import pandas as pd


def generate_synthetic_data(
    n: int = 100,
    mean: float = 50.0,
    std: float = 10.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Genera un DataFrame con una variable numerica sintetica
    usando una distribucion normal.
    """
    rng = np.random.default_rng(seed)

    data = rng.normal(loc=mean, scale=std, size=n)

    df = pd.DataFrame({
        "variable_1": data
    })

    return df
