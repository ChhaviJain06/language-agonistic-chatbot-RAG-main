from server.rag.citations import extract_ids, number_citations, strip_tags
from server.rag.retrieve import Hit


def hit(cid, article="Ashoka"):
    url = f"https://en.wikipedia.org/wiki/{article.replace(' ', '_')}"
    return Hit(cid, article, "Reign", f"text of {cid}", 0.9, url)


def test_tags_become_numbered_markers_in_order_of_first_use():
    hits = [hit("ashoka::0001"), hit("kalinga::0002", "Kalinga War")]
    answer = "He fought [kalinga::0002] and ruled [ashoka::0001]. Later [kalinga::0002]."
    r = number_citations(answer, hits)
    assert r.text == "He fought [1] and ruled [2]. Later [1]."
    assert [s["chunk_id"] for s in r.sources] == ["kalinga::0002", "ashoka::0001"]
    assert r.sources[0]["url"].endswith("Kalinga_War") and r.invalid == 0


def test_fabricated_ids_are_dropped_never_shown():       # C8
    r = number_citations("Claim [ashoka::0001] and fake [made_up::9999].", [hit("ashoka::0001")])
    assert "made_up" not in r.text and r.invalid == 1 and len(r.sources) == 1


def test_multi_id_tags_and_graph_tag():
    r = number_citations("Fact [a::1, b::2] [graph].", [hit("a::1"), hit("b::2")])
    assert r.text == "Fact [1][2]." and r.invalid == 0


def test_answer_without_citations_has_no_sources():
    r = number_citations("I do not know.", [hit("a::1")])
    assert r.sources == [] and r.text == "I do not know."


def test_strip_tags_and_extract_ids():
    assert strip_tags("Ashoka【ashoka::0001】 ruled [ashoka::0002] well [graph].") == \
        "Ashoka ruled well."
    assert strip_tags("Keep [this] and [1].") == "Keep [this] and [1]."
    assert extract_ids("x [a::1] y [b::2, c::3]") == ["a::1", "b::2", "c::3"]
