import os
from neo4j import GraphDatabase


class Neo4jConnector:
    """Manages the connection and query execution against Neo4j."""

    def __init__(self):
        uri = os.getenv("NEO4J_URI", "")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # Verify connectivity at startup
        self.driver.verify_connectivity()

    def run_query(self, cypher: str, params: dict = None) -> list:
        """
        Executes a Cypher query and returns all records as a list of dicts.
        """
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]

    def close(self):
        """Closes the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
