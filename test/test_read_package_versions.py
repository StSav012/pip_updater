from importlib.metadata import Distribution

from pip_updater import read_package_versions


def test_read_package_versions() -> None:
    print(read_package_versions(Distribution.from_name("pip_updater")))


if __name__ == "__main__":
    test_read_package_versions()
