from neo4j import GraphDatabase

from config import cfg


if not cfg.NEO4J_URI:
    raise ValueError("NEO4J_URI is missing in .env")
if not cfg.NEO4J_USERNAME:
    raise ValueError("NEO4J_USERNAME is missing in .env")
if not cfg.NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD is missing in .env")


driver = GraphDatabase.driver(
    cfg.NEO4J_URI,
    auth=(cfg.NEO4J_USERNAME, cfg.NEO4J_PASSWORD),
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