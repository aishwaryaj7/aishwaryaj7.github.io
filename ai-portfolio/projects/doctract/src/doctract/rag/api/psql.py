import psycopg2
from llama_index.vector_stores.postgres import PGVectorStore
from doctract.constants import *


def recreate_postgres_database(database_name: str) -> None:
    """
    Drops the target database if it exists and recreates it.
    This function connects to the default 'postgres' database
    to perform admin-level operations.

    Args:
        database_name (str): Name of the database to recreate.
    """
    connection = psycopg2.connect(
        dbname="postgres",
        host=POSTGRES_HOST,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        port=POSTGRES_PORT,
    )
    connection.autocommit = True

    with connection.cursor() as cursor:
        cursor.execute(f"DROP DATABASE IF EXISTS {database_name}")
        cursor.execute(f"CREATE DATABASE {database_name}")
    connection.close()


def initialize_pgvector_store() -> PGVectorStore:
    """
    Initializes a PGVectorStore instance using provided parameters.

    Args:
        database_name (str): The target PostgreSQL database name.
        table_name (str): The table to store vector embeddings.
        embedding_dim (int): The dimension of the vector embeddings.

    Returns:
        PGVectorStore: Configured PGVectorStore instance.
    """
    # recreate_postgres_database(TARGET_DATABASE_NAME)
    vector_store_instance = PGVectorStore.from_params(
        database=TARGET_DATABASE_NAME,
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        table_name=TARGET_TABLE_NAME,
        embed_dim=EMBEDDING_DIMENSION,
    )

    return vector_store_instance