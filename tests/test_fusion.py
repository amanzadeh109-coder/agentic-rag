from app.retrieval import rrf


def test_item_in_both_lists_wins():
    assert rrf([["a", "b", "c"], ["c", "d"]])[0] == "c"


def test_all_items_kept_once():
    out = rrf([["a", "b"], ["b", "c"]])
    assert sorted(out) == ["a", "b", "c"]


def test_empty():
    assert rrf([[], []]) == []