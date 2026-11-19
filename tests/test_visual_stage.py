from server.providers.base import GraphHit
from server.visual_stage import build_subgraph, extract_visual_payload


def test_nothing_matches_means_nothing_is_invented():
    p = extract_visual_payload("What is the weather?", "I don't know.", [], [])
    assert p.matched is False and p.entity is None and p.era_years is None
    assert p.geo_coordinates is None and p.citation is None
    assert p.subgraph == {"nodes": [], "links": []}          # no fabricated ERA_OF edge


def test_known_entity_gets_map_era_and_citation():
    src = [{"chunk_id": "plassey::0001", "article": "Battle of Plassey", "section": "Battle",
            "text": "x" * 500, "url": "u"}]
    p = extract_visual_payload("When was the Battle of Plassey?", "In 1757 [1].", [], [], src)
    assert p.matched and p.entity == "Plassey" and p.era_years == [1757, 1757]
    assert p.geo_coordinates == [23.80, 88.25]
    assert p.citation["chunk_id"] == "plassey::0001" and len(p.citation["text"]) == 400


def test_longest_key_wins_and_word_boundaries_hold():
    assert extract_visual_payload("Tell me about Rajendra Chola", "", [], []).entity == \
        "Rajendra Chola"
    assert extract_visual_payload("Who is Mughalsarai?", "", [], []).entity is None


def test_edge_direction_is_preserved():
    forward = GraphHit([{"entity": "Shah Jahan"}, {"rel": "FATHER_OF", "dir": "out"},
                        {"entity": "Aurangzeb"}], ["a::1"], 1.0)
    backward = GraphHit([{"entity": "Aurangzeb"}, {"rel": "FATHER_OF", "dir": "in"},
                         {"entity": "Shah Jahan"}], ["a::1"], 1.0)
    for hit in (forward, backward):
        g = build_subgraph([hit])
        assert g["links"] == [{"source": "Shah Jahan", "target": "Aurangzeb",
                               "relation": "FATHER_OF"}]
    assert len(build_subgraph([forward, backward])["links"]) == 1     # deduplicated
