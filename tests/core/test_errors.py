from docling_rag.core.errors import cause_chain


def test_cause_chain_yields_exception_itself():
    e = ValueError("solo")
    assert list(cause_chain(e)) == [e]


def test_cause_chain_walks_cause_and_context():
    root = ValueError("root")
    mid = KeyError("mid")
    mid.__cause__ = root
    top = RuntimeError("top")
    top.__context__ = mid  # __cause__ пуст → идём по __context__
    assert [type(x) for x in cause_chain(top)] == [RuntimeError, KeyError, ValueError]


def test_cause_chain_prefers_cause_over_context():
    cause, context = ValueError("cause"), KeyError("context")
    top = RuntimeError("top")
    top.__cause__ = cause
    top.__context__ = context
    chain = list(cause_chain(top))
    assert cause in chain and context not in chain


def test_cause_chain_cycle_safe():
    a, b = ValueError("a"), KeyError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert len(list(cause_chain(a))) == 2
