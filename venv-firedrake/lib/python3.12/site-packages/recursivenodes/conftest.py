
import pytest


@pytest.fixture(scope="session", autouse=True)
def init(doctest_namespace):
    from recursivenodes import recursive_nodes
    doctest_namespace['recursive_nodes'] = recursive_nodes
