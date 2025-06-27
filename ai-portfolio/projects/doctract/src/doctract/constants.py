import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

CWD = Path().cwd()
DATA_DIR = CWD.joinpath("data")

POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
TARGET_DATABASE_NAME = os.getenv("VECTOR_DB_NAME")
TARGET_TABLE_NAME = os.getenv("VECTOR_TABLE_NAME")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION"))