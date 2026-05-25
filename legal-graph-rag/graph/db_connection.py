import os
from neo4j import GraphDatabase
from dotenv import load_dotenv


load_dotenv()


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


if not NEO4J_URI:
    raise ValueError("NEO4J_URI is missing in .env")

if not NEO4J_USERNAME:
    raise ValueError("NEO4J_USERNAME is missing in .env")

if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD is missing in .env")


driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)


def get_driver():
    return driver


def close_driver():
    driver.close()


def test_connection():
    with driver.session() as session:
        result = session.run("RETURN 'Neo4j Connected Successfully!' AS message")
        print(result.single()["message"])


if __name__ == "__main__":
    test_connection()