import sys

from pip_updater import list_packages


def test_list_packages() -> None:
    for p in list_packages([sys.exec_prefix]):
        print(p.name, p)


if __name__ == "__main__":
    test_list_packages()
