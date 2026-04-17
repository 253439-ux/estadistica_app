import os
from typing import Optional

from google import genai


def build_stats_prompt(summary: dict) -> str:
    """
    Construye un prompt interno para interpretar una prueba Z.
    """
    return f"""
Actua como un asistente estadistico riguroso y claro.

Analiza los siguientes resultados de una prueba Z para una muestra:

- Variable analizada: {summary['variable_name']}
- Tamano de muestra (n): {summary['n']}
- Media muestral: {summary['sample_mean']:.4f}
- Media bajo H0 (mu_0): {summary['mu_0']:.4f}
- Desviacion estandar poblacional conocida (sigma): {summary['sigma']:.4f}
- Tipo de prueba: {summary['tail_type']}
- Nivel de significancia (alpha): {summary['alpha']:.4f}
- Estadistico Z: {summary['z_stat']:.4f}
- p-value: {summary['p_value']:.6f}
- Valor critico: {summary['critical_value']:.4f}
- Decision automatica: {summary['decision']}

Tu tarea:
1. Explica brevemente que significa el resultado.
2. Indica claramente si se rechaza o no la hipotesis nula.
3. Comenta si el supuesto de usar prueba Z parece razonable, considerando que:
   - se asume sigma conocida
   - n debe ser al menos 30
4. No inventes datos.
5. Responde en espanol claro y en un tono academico breve.
""".strip()


def ask_gemini(summary: dict, api_key: Optional[str] = None) -> str:
    """
    Envia el resumen estadistico a Gemini y devuelve una interpretacion textual.
    """
    final_api_key = api_key or os.getenv("GEMINI_API_KEY")

    if not final_api_key:
        raise ValueError("No se proporciono una API key de Gemini.")

    client = genai.Client(api_key=final_api_key)
    prompt = build_stats_prompt(summary)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text if response.text else "No se recibio respuesta de Gemini."