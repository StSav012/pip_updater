from pip_updater import orphaned_packages


def test_orphaned_packages() -> None:
    print(orphaned_packages())


if __name__ == "__main__":
    test_orphaned_packages()
