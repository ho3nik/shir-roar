import os

import pytest

from shirroar.trace import Request

HERE = os.path.dirname(os.path.abspath(__file__))
MOONCAKE = os.environ.get("MOONCAKE_TRACE", os.path.join(HERE, "..", "data", "mooncake_trace.jsonl"))


def chain(prefix, n, start):
    """Blocks of a request: an existing prefix plus n fresh ids starting at `start`."""
    return tuple(prefix) + tuple(range(start, start + n))


@pytest.fixture
def toy_requests():
    # Three requests share block 0; two of them also share block 1. Prefix hashing holds.
    return [
        Request(0.0, (0, 1, 2), 1536, 10),
        Request(1.0, (0, 1, 3), 1536, 10),
        Request(2.0, (0, 4), 1024, 10),
        Request(3.0, (0, 1, 2, 5), 2048, 10),
    ]


@pytest.fixture(scope="session")
def mooncake():
    if not os.path.exists(MOONCAKE):
        pytest.skip("Mooncake trace not present; run scripts/fetch_mooncake.sh")
    from shirroar.trace import load_mooncake
    return load_mooncake(MOONCAKE)
