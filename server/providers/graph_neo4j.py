"""Neo4j-backed multi-hop entity graph. Additive only (C11): health() and query_entity()
both degrade to "nothing found" on any failure, never raise — graph_retrieve.py's caller
must always get a valid answer from vector retrieval alone.
"""

from __future__ import annotations

import logging

from server.config import get_settings
from server.providers.base import GraphHit, GraphProvider

logger = logging.getLogger(__name__)

# Undirected match (a relation is useful whichever way the question phrases it), but the
# direction of every hop is recorded in the result so "A FATHER_OF B" is never confused
# with "B FATHER_OF A". Strongest, shortest evidence first.
_QUERY = """
MATCH path = (a:Entity)-[*1..%d]-(b:Entity)
WHERE a.key = $key OR a.key STARTS WITH $key
RETURN path
ORDER BY reduce(c = 1.0, r IN relationships(path) |
                CASE WHEN coalesce(r.confidence, 1.0) < c THEN coalesce(r.confidence, 1.0)
                     ELSE c END) DESC,
         length(path) ASC
LIMIT 20
"""


class Neo4jGraphProvider(GraphProvider):
    name = "neo4j"

    def __init__(self, uri: str | None = None, user: str | None = None,
                 password: str | None = None):
        s = get_settings()
        self._uri = uri or s.neo4j_uri
        self._user = user or s.neo4j_user
        self._password = password or s.neo4j_password
        self._driver = None

    @property
    def driver(self):
        """Public on purpose: build_graph.py needs raw write access (MERGE entities and
        relations) that query_entity()'s read-only interface doesn't expose."""
        if self._driver is None:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                self._uri, auth=(self._user, self._password),
                connection_timeout=3, max_connection_lifetime=600)
        return self._driver

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def health(self) -> bool:
        try:
            async with self.driver.session() as session:
                result = await session.run("RETURN 1")
                await result.consume()
            return True
        except Exception as e:  # noqa: BLE001 - C11: never raise, always degrade
            logger.warning("neo4j unreachable: %r", e)
            return False

    async def query_entity(self, entity_name: str, max_hops: int = 2) -> list[GraphHit]:
        hops = max(1, min(int(max_hops), 3))
        try:
            hits: list[GraphHit] = []
            async with self.driver.session() as session:
                result = await session.run(_QUERY % hops, key=entity_name.strip().lower())
                async for record in result:
                    hits.append(_path_to_hit(record["path"]))
            return hits
        except Exception as e:  # noqa: BLE001 - C11: unreachable graph -> no hits, never a 500
            logger.warning("neo4j query failed for %r: %r", entity_name, e)
            return []


def _path_to_hit(path) -> GraphHit:
    """Converts a neo4j Path into ordered steps, keeping each relation's direction."""
    steps: list[dict] = [{"entity": path.nodes[0].get("name")}]
    chunk_ids: set[str] = set()
    confidences: list[float] = []
    prev = path.nodes[0]
    for rel, node in zip(path.relationships, path.nodes[1:], strict=True):
        forward = rel.start_node.element_id == prev.element_id
        steps.append({"rel": rel.type, "dir": "out" if forward else "in"})
        steps.append({"entity": node.get("name")})
        chunk_ids.update(rel.get("source_chunk_ids") or [])
        confidences.append(rel.get("confidence", 1.0))
        prev = node
    return GraphHit(
        path=steps,
        source_chunk_ids=sorted(chunk_ids),
        confidence=min(confidences) if confidences else 1.0,
    )


class NullGraphProvider(GraphProvider):
    """GRAPH_PROVIDER=none. Always unhealthy, so callers' C11 fallback path (vector-only)
    runs with zero special-casing for "graph disabled"."""

    name = "none"

    async def health(self) -> bool:
        return False

    async def query_entity(self, entity_name: str, max_hops: int = 2) -> list[GraphHit]:
        return []
