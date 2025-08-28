from operator import eq, le, gt, ge, lt, ne

# noinspection PyProtectedMember
from pip_updater import compare_versions


def test_compare_versions() -> None:
    assert compare_versions("1.2", "1.1", lt) == False
    assert compare_versions("1.2", "1.1", le) == False
    assert compare_versions("1.2", "1.1", eq) == False
    assert compare_versions("1.2", "1.1", ge) == True
    assert compare_versions("1.2", "1.1", gt) == True
    assert compare_versions("1.2", "1.1", ne) == True

    assert compare_versions("1.2", "1.20", lt) == True
    assert compare_versions("1.2", "1.20", le) == True
    assert compare_versions("1.2", "1.20", eq) == False
    assert compare_versions("1.2", "1.20", ge) == False
    assert compare_versions("1.2", "1.20", gt) == False
    assert compare_versions("1.2", "1.20", ne) == True

    assert compare_versions("1.2", "1.20a", lt) == False
    assert compare_versions("1.2", "1.20a", le) == False
    assert compare_versions("1.2", "1.20a", eq) == False
    assert compare_versions("1.2", "1.20a", ge) == True
    assert compare_versions("1.2", "1.20a", gt) == True
    assert compare_versions("1.2", "1.20a", ne) == True

    assert compare_versions("1.2", "1.2.3", lt) == True
    assert compare_versions("1.2", "1.2.3", le) == True
    assert compare_versions("1.2", "1.2.3", eq) == False
    assert compare_versions("1.2", "1.2.3", ge) == False
    assert compare_versions("1.2", "1.2.3", gt) == False
    assert compare_versions("1.2", "1.2.3", ne) == True

    assert compare_versions("1.2.3a", "1.2.3b", lt) == True
    assert compare_versions("1.2.3a", "1.2.3b", le) == True
    assert compare_versions("1.2.3a", "1.2.3b", eq) == False
    assert compare_versions("1.2.3a", "1.2.3b", ge) == False
    assert compare_versions("1.2.3a", "1.2.3b", gt) == False
    assert compare_versions("1.2.3a", "1.2.3b", ne) == True

    assert compare_versions("1.2.3.a", "1.2.3.b", lt) == True
    assert compare_versions("1.2.3.a", "1.2.3.b", le) == True
    assert compare_versions("1.2.3.a", "1.2.3.b", eq) == False
    assert compare_versions("1.2.3.a", "1.2.3.b", ge) == False
    assert compare_versions("1.2.3.a", "1.2.3.b", gt) == False
    assert compare_versions("1.2.3.a", "1.2.3.b", ne) == True

    assert compare_versions("1.2.3.1.a", "1.2.3.b", lt) == False
    assert compare_versions("1.2.3.1.a", "1.2.3.b", le) == False
    assert compare_versions("1.2.3.1.a", "1.2.3.b", eq) == False
    assert compare_versions("1.2.3.1.a", "1.2.3.b", ge) == True
    assert compare_versions("1.2.3.1.a", "1.2.3.b", gt) == True
    assert compare_versions("1.2.3.1.a", "1.2.3.b", ne) == True

    assert compare_versions("1.2", "1.2", eq) == True
    assert compare_versions("1.2", "1.2", ne) == False

    assert compare_versions("1.2.3a", "1.2.3a", eq) == True
    assert compare_versions("1.2.3a", "1.2.3a", ne) == False


if __name__ == "__main__":
    test_compare_versions()
