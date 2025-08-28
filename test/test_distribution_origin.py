# noinspection PyProtectedMember
from pip_updater import Distribution


def test_distribution_origin() -> None:
    assert Distribution.from_name("pip_updater").origin is None


if __name__ == "__main__":
    test_distribution_origin()
