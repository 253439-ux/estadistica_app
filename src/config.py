from dotenv import load_dotenv


def load_environment() -> None:
    """
    Carga las variables de entorno desde el archivo .env.
    """
    load_dotenv()
