import os
from typing import Dict, Iterable, List

try:
    from neo4j import GraphDatabase  # type: ignore[reportMissingImports]
except Exception:
    GraphDatabase = None


class GraphStore:
    """Small Neo4j adapter for storing and traversing entity relations."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._driver = None
        self.last_error = ""

        if not enabled:
            return

        if GraphDatabase is None:
            self.enabled = False
            return

        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD")

        if not password:
            self.enabled = False
            self.last_error = "NEO4J_PASSWORD is missing"
            return

        candidate_uris = [uri]
        if "localhost" in uri or "127.0.0.1" in uri:
            candidate_uris.append("bolt://neo4j:7687")
        if "neo4j" in uri:
            candidate_uris.append("bolt://localhost:7687")

        seen = set()
        for candidate_uri in candidate_uris:
            if candidate_uri in seen:
                continue
            seen.add(candidate_uri)
            try:
                driver = GraphDatabase.driver(candidate_uri, auth=(user, password))
                driver.verify_connectivity()
                self._driver = driver
                self.enabled = True
                self.last_error = ""
                return
            except Exception as e:
                self.last_error = f"Unable to connect to Neo4j at {candidate_uri}: {e}"

        self.enabled = False
        self._driver = None

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def reset_graph(self) -> None:
        if not self.enabled or self._driver is None:
            return
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    @staticmethod
    def _sanitize_triplet(triplet: Dict) -> Dict | None:
        head = str(triplet.get("head", "")).strip()
        relation = str(triplet.get("relation", "")).strip()
        tail = str(triplet.get("tail", "")).strip()

        if not head or not relation or not tail:
            return None

        return {
            "head": head,
            "relation": relation,
            "tail": tail,
            "source": str(triplet.get("source", "")).strip(),
            "chunk_id": str(triplet.get("chunk_id", "")).strip(),
            "confidence": float(triplet.get("confidence", 0.0) or 0.0),
        }

    def upsert_triplets(self, triplets: Iterable[Dict]) -> Dict[str, int]:
        if not self.enabled or self._driver is None:
            return {"entities": 0, "relations": 0}

        valid_triplets: List[Dict] = []
        for triplet in triplets:
            clean_triplet = self._sanitize_triplet(triplet)
            if clean_triplet:
                valid_triplets.append(clean_triplet)

        if not valid_triplets:
            return {"entities": 0, "relations": 0}

        query = """
        UNWIND $triplets AS t
        MERGE (h:Entity {name: t.head})
        MERGE (ta:Entity {name: t.tail})
        MERGE (h)-[r:RELATED {relation: t.relation, source: t.source, chunk_id: t.chunk_id}]->(ta)
        SET r.confidence = t.confidence
        """

        with self._driver.session() as session:
            session.run(query, triplets=valid_triplets)

        entities = {t["head"] for t in valid_triplets}.union({t["tail"] for t in valid_triplets})
        return {"entities": len(entities), "relations": len(valid_triplets)}

    def get_neighbors(self, seed_entities: List[str], hops: int = 1, limit: int = 20) -> List[Dict]:
        if not self.enabled or self._driver is None:
            return []

        cleaned = [e.strip() for e in seed_entities if e and e.strip()]
        if not cleaned:
            return []

        cypher = """
        MATCH (s:Entity)
        WHERE s.name IN $seed_entities
        MATCH p=(s)-[:RELATED*1..$hops]-(n:Entity)
        UNWIND relationships(p) AS rel
        WITH DISTINCT startNode(rel) AS h, rel, endNode(rel) AS t
        RETURN h.name AS head, rel.relation AS relation, t.name AS tail,
               coalesce(rel.source, '') AS source,
               coalesce(rel.chunk_id, '') AS chunk_id,
               coalesce(rel.confidence, 0.0) AS confidence
        LIMIT $limit
        """

        with self._driver.session() as session:
            result = session.run(
                cypher,
                seed_entities=cleaned,
                hops=max(1, int(hops)),
                limit=max(1, int(limit)),
            )
            return [record.data() for record in result]
