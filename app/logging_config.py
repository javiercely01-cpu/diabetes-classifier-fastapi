"""
Configuración centralizada de logging para la API.

Escribe en consola y en logs/api.log con un formato que permite
monitorear: timestamp, nivel, módulo y mensaje (request_id, entrada,
predicción, código de estado y latencia de cada solicitud).
"""
import logging
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "api.log")


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("diabetes_api")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger  # evita duplicar handlers si se llama más de una vez

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
