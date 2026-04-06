from __future__ import annotations

import argparse
from getpass import getpass

from neo4j import GraphDatabase

from settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clear all data from the local Neo4j database.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    return parser.parse_args()


def confirm() -> bool:
    print("This will delete all nodes and relationships from Neo4j.")
    response = getpass('Type CLEAR to continue: ')
    return response.strip().upper() == "CLEAR"


def main() -> int:
    args = parse_args()
    if not args.yes and not confirm():
        print("Aborted.")
        return 1

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            stats = session.run(
                "MATCH (n) OPTIONAL MATCH ()-[r]->() RETURN count(DISTINCT n) AS node_count, count(r) AS relationship_count"
            ).single()
            node_count = stats["node_count"] if stats else 0
            relationship_count = stats["relationship_count"] if stats else 0
            session.run("MATCH (n) DETACH DELETE n").consume()
        print(
            f"Cleared Neo4j graph data. Deleted {node_count} node(s) and {relationship_count} relationship(s)."
        )
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())