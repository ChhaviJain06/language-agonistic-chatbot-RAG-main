from server.session import MAX_HISTORY_MESSAGES, SessionState, apology_for


def test_history_window_and_heard_text_only():
    s = SessionState(room="r")
    for i in range(5):
        s.remember(f"q{i}", f"a{i}")
    assert len(s.history) == MAX_HISTORY_MESSAGES and s.history[-1]["content"] == "a4"
    s.remember("interrupted", "")                # nothing heard -> only the question is kept
    assert s.history[-1] == {"role": "user", "content": "interrupted"}


def test_apology_matches_language():
    assert "क्षमा" in apology_for("hi") and "Sorry" in apology_for("en")
    assert "kuch gadbad" in apology_for("hi-en") and "Sorry" in apology_for("xx")
