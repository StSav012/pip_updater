def main() -> None:
    from inspect import Signature, signature
    from pathlib import Path
    from typing import Any, Collection

    import pip_updater

    __all__: Collection[str] = getattr(pip_updater, "__all__", [])

    root: Path = Path(__file__).parent

    item: str
    for item in __all__:
        o: Any = getattr(pip_updater, item)
        if not isinstance(o, type) and callable(o):
            s: Signature = signature(o)
            if not s.parameters:
                with open(root / f"test_{item}.py", "wt") as f_out:
                    f_out.write(
                        "\n".join(
                            (
                                f"from pip_updater import {item}",
                                "",
                                "",
                                f"def test_{item}() -> None:",
                                (
                                    f"    {item}()"
                                    if s.return_annotation is None
                                    else f"    print({item}())"
                                ),
                                "",
                                "",
                                'if __name__ == "__main__":',
                                f"    test_{item}()",
                                "",
                            )
                        )
                    )


if __name__ == "__main__":
    main()
