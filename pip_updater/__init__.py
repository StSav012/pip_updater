import argparse
import operator
import os
import platform
import re
import site
import sys
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from http.client import HTTPResponse
from importlib.metadata import Distribution, DistributionFinder
from itertools import zip_longest
from pathlib import Path
from shutil import which
from ssl import SSLCertVerificationError, SSLContext
from subprocess import call, check_output
from typing import (
    Callable,
    Final,
    Iterable,
    Iterator,
    Literal,
    Protocol,
    Sequence,
)
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

try:
    from subprocess import DETACHED_PROCESS
except (ModuleNotFoundError, ImportError):
    # noinspection PyFinal
    DETACHED_PROCESS = 0

__all__ = [
    "Graph",
    "read_package_versions",
    "list_packages",
    "list_packages_tree",
    "update_packages",
    "update_package",
    "orphaned_packages",
    "print_orphaned_packages",
]

# backport the functions to Python<3.13
if not hasattr(Distribution, "_load_json"):
    # noinspection PyUnresolvedReferences,PyProtectedMember
    from importlib.metadata._functools import pass_none
    from json import loads
    from types import SimpleNamespace

    Distribution._load_json = lambda self, filename: pass_none(loads)(
        self.read_text(filename),
        object_hook=lambda data: SimpleNamespace(**data),
    )
if not hasattr(Distribution, "origin"):
    # noinspection PyPropertyAccess,PyProtectedMember
    Distribution.origin = property(
        lambda self: Distribution._load_json(self, "direct_url.json")
    )


PIP: Final[str] = "pip"
UV: Final[str] = "uv"
UV_CMD: Final[str | None] = which(UV)

GIT: Final[str] = "git"
# Use the `{url}` placeholder to provide a package URL to the VCS
VCS_VERSION_TEMPLATES: dict[str, Sequence[str]] = {}
if git := which(GIT):
    VCS_VERSION_TEMPLATES[GIT] = [git, "ls-remote", "--heads", "{url}"]


class Graph:
    class Node:
        def __init__(self, value: str) -> None:
            self.value: str = value
            self.children: set[Graph.Node] = set()
            self.parents: set[Graph.Node] = set()

        def __repr__(self) -> str:
            return (
                f"{self.value!r}: "
                + "{"
                + ", ".join(repr(child) for child in self.children)
                + "}"
            )

        def __str__(self) -> str:
            return self.value

        def __eq__(self, other: object) -> bool:
            if isinstance(other, str):
                return self.value == other
            if isinstance(other, Graph.Node):
                return self.value == other.value
            return NotImplemented

        def __hash__(self) -> int:
            return hash(self.value)

        def __getitem__(self, item: str) -> "Graph.Node":
            if self == item:
                return self
            node: Graph.Node
            for node in self.children:
                if node == item:
                    return node
            raise IndexError(f"Item {item} not found")

    def __init__(self) -> None:
        self.nodes: set[Graph.Node] = set()

    def find(self, value: str) -> Node | None:
        def recursive_find(where: Iterable[Graph.Node]) -> Graph.Node | None:
            node: Graph.Node
            for node in where:
                if node == value:
                    return node
                found: None | Graph.Node = recursive_find(node.children)
                if found is not None:
                    return found
            return None

        return recursive_find(self.nodes)

    def add_node(self, parent: str | Node, value: str | None = None) -> None:
        if isinstance(parent, str):
            found_parent: Graph.Node | None = self.find(parent)
            if found_parent is None:
                parent = Graph.Node(parent)
                self.nodes.add(parent)
            else:
                parent = found_parent
        if value is not None:
            found_value: Graph.Node | None = self.find(value)
            node: Graph.Node
            if found_value is None:
                node = Graph.Node(value)
            else:
                if found_value in self.nodes:
                    self.nodes.remove(found_value)
                for parent in found_value.parents:
                    parent.children.remove(found_value)
                node = found_value
            node.parents.add(parent)
            parent.children.add(node)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "(\n"
            + ",\n".join(repr(node) for node in self.nodes)
            + ")"
        )

    def __invert__(self) -> "Graph":
        graph: Graph = Graph()

        def recursive_children(child_node: Graph.Node) -> None:
            def recursive_parents(
                parent_node: Graph.Node,
                parents: set[Graph.Node],
            ) -> None:
                if not parents:
                    return
                parent: Graph.Node
                for parent in parents:
                    parent_node.children.add(Graph.Node(parent.value))
                parents = parent_node.children.copy()
                while parents:
                    parent = parents.pop()
                    recursive_parents(parent_node[parent.value], parent.parents)

            if not child_node.children:
                new_node: Graph.Node = Graph.Node(child_node.value)
                new_node.children = set(
                    Graph.Node(parent.value) for parent in child_node.parents
                )
                recursive_parents(child_node, child_node.parents)
                graph.nodes.add(new_node)
            else:
                child: Graph.Node
                for child in child_node.children:
                    recursive_children(child)

        node: Graph.Node
        for node in self.nodes:
            recursive_children(node)
        return graph

    def top(self) -> frozenset[str]:
        return frozenset(node.value for node in self.nodes)


def is_python_version_equal(version: str) -> bool:
    return all(
        gv == "*" or gv == str(pv)
        for gv, pv in zip(version.split("."), sys.version_info)
    )


def is_python_version_less(version: str) -> bool:
    for gv, pv in zip(version.split("."), sys.version_info):
        if gv == "*":
            break
        if gv.isdecimal() and isinstance(pv, int):
            if int(gv) != pv:
                return int(gv) > pv
        else:
            # FIXME: that's not how it should be done
            if gv != str(pv):
                return gv > str(pv)
    return False


def is_python_version_greater(version: str) -> bool:
    for gv, pv in zip(version.split("."), sys.version_info):
        if gv == "*":
            break
        if gv.isdecimal() and isinstance(pv, int):
            if int(gv) != pv:
                return int(gv) < pv
        else:
            # FIXME: that's not how it should be done
            if gv != str(pv):
                return gv < str(pv)
    return False


def compare_versions(
    version_1: str, version_2: str, cmp: Callable[[int | str, int | str], bool]
) -> bool:
    for v1, v2 in zip_longest(version_1.split("."), version_2.split("."), fillvalue=""):
        if v1 == v2:
            continue
        if v1.isdecimal() and v2.isdecimal():
            if int(v1) != int(v2):
                return cmp(int(v1), int(v2))
        elif v1.isdecimal() and not v2.isdecimal():
            return cmp(int(v1), 0)
        elif not v1.isdecimal() and v2.isdecimal():
            return cmp(0, int(v2))
        else:
            return cmp(v1, v2)
    return cmp(version_1, version_2)


class PackageFileParser(HTMLParser):
    def __init__(self, package_name: str, *, pre: bool = False) -> None:
        super().__init__()
        self._package_name: str = package_name.replace("-", "_").casefold()
        self._path: deque[str] = deque()
        self._versions: deque[str] = deque()
        self._pre: bool = pre
        self._requires_python: str = ""
        self._yanked: str = ""

    def _is_python_version_valid(self) -> bool:
        for version_condition in self._requires_python.split(","):
            version_condition = version_condition.strip()
            if version_condition.startswith(">="):
                if is_python_version_less(version_condition[2:]):
                    return False
            elif version_condition.startswith(">"):
                if is_python_version_less(
                    version_condition[1:]
                ) or is_python_version_equal(version_condition[1:]):
                    return False
            if version_condition.startswith("<="):
                if is_python_version_greater(version_condition[2:]):
                    return False
            elif version_condition.startswith("<"):
                if is_python_version_greater(
                    version_condition[1:]
                ) or is_python_version_equal(version_condition[1:]):
                    return False
            if version_condition.startswith("!="):
                if is_python_version_equal(version_condition[2:]):
                    return False
            if version_condition.startswith("=="):
                if not is_python_version_equal(version_condition[2:]):
                    return False
            if version_condition.startswith("~="):
                pass  # do not know what to do
        return True

    @staticmethod
    def _is_arch_valid(arch: str) -> bool:
        if arch == "any":
            return True
        if sys.platform.startswith("win"):
            if platform.machine().endswith("64"):
                return f"win_{platform.machine()}" in arch
            else:
                return sys.platform in arch
        elif sys.platform.startswith("darwin"):
            if platform.machine().endswith("64"):
                return "macosx" in arch and platform.machine() in arch
            else:
                return "macosx" in arch and ("intel" in arch or "universal" in arch)
        elif sys.platform.startswith("linux"):
            return sys.platform in arch and platform.machine() in arch
        # if nothing matches, the arch might still be valid
        return True

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._path.append(tag)
        attrs_dict: dict[str, str | None] = dict(attrs)
        self._requires_python = attrs_dict.get("data-requires-python", "") or ""
        self._yanked = attrs_dict.get("data-yanked", "") or ""

    def handle_endtag(self, tag: str) -> None:
        while self._path and self._path.pop() != tag:
            pass

    def handle_data(self, data: str) -> None:
        if self._path == deque(["html", "body", "a"]):
            if self._yanked:
                return
            if not self._is_python_version_valid():
                return
            data = data.casefold()
            valid_suffixes: tuple[str, ...] = (
                ".whl",
                ".tar.bz2",
                ".tar.gz",
                ".tgz",
                ".zip",
            )
            silently_invalid_suffixes: tuple[str, ...] = (
                ".exe",
                ".msi",
                ".egg",
                ".noarch.rpm",
                ".src.rpm",
            )
            if data.endswith(valid_suffixes):
                for suffix in valid_suffixes:
                    data = data.removesuffix(suffix)
            elif data.endswith(silently_invalid_suffixes):
                return
            else:
                print(f"Odd filename found: {data!r}", file=sys.stderr)
                return
            parts: Sequence[str] = data.removeprefix(f"{self._package_name}-").split(
                "-"
            )
            if parts:
                version, *parts = parts
                last_version_part: str = version.split(".")[-1]
                if not self._pre and not (
                    last_version_part.isdecimal()
                    or last_version_part.startswith("post")
                ):
                    # likely ends with “rc,” “dev,” “a,” or “b”
                    return
                if not parts or self._is_arch_valid(parts[-1]):
                    self._versions.append(version)

    @property
    def versions(self) -> deque[str]:
        return self._versions


def read_package_versions(
    distribution: Distribution,
    pre: bool = False,
) -> Sequence[str]:
    def read_package_versions_pip() -> Sequence[str]:
        r: HTTPResponse
        parser: PackageFileParser = PackageFileParser(distribution.name, pre=pre)
        try:
            with urlopen(
                f"https://pypi.org/simple/{distribution.name.replace('_', '-')}/"
            ) as r:
                parser.feed(r.read().decode())
        except HTTPError as ex:
            if ex.code == 404:
                print(f"{distribution.name} not found", file=sys.stderr)
            else:
                print(f"{distribution.name}: {ex!s}", file=sys.stderr)
        except URLError as ex:
            if isinstance(ex.reason, SSLCertVerificationError):
                print("CRITICAL: SSL Certificate Verification failed.", end=" ")
                while (
                    decision := input("Continue insecurely? [y|n]").casefold()
                ) not in ("y", "n"):
                    pass
                if decision == "y":
                    with urlopen(
                        f"https://pypi.org/simple/{distribution.name.replace('_', '-')}/",
                        context=SSLContext(),
                    ) as r:
                        parser.feed(r.read().decode())
                else:
                    print(f"Skipping {distribution.name}", file=sys.stderr)
                    return []
            else:
                raise
        return parser.versions

    def read_package_versions_vcs() -> Sequence[str]:
        return [
            check_output(
                args=[arg.format(url=url) for arg in VCS_VERSION_TEMPLATES[vcs]],
            )
            .split()[0]
            .decode()
        ]

    if distribution.origin is not None:
        try:
            vcs: str = distribution.origin.vcs_info.vcs
        except AttributeError:
            if "archive_info" in dir(distribution.origin):
                print(f"{distribution.name} is installed from a local source")
                return []
            print(f"{distribution.name} came from an unknown source")
        else:
            try:
                url: str = distribution.origin.url
            except AttributeError:
                print(f"{distribution.name} came from an unsupported source")
            else:
                if vcs in VCS_VERSION_TEMPLATES:
                    return read_package_versions_vcs()
                print(f"{distribution.name} is installed directly from {url}")

    return read_package_versions_pip()


def update_package(
    distribution: Distribution,
    executable: str | os.PathLike[str] | None = None,
    use_pure_pip: bool = False,
    latest_version: str = "",
    creationflags: int = 0,
) -> int | None:
    if not executable:
        executable = sys.executable

    package_description: str = distribution.name
    try:
        url: str = distribution.origin.url
        vcs: str = distribution.origin.vcs_info.vcs
        commit_id: str = distribution.origin.vcs_info.commit_id
        package_description = vcs + "+" + url + "@" + commit_id
    except AttributeError:
        if latest_version:
            package_description += "==" + latest_version

    args: list[str] = (
        [UV_CMD, PIP, "install", "--python", executable, "-U"]
        if UV_CMD is not None
        and not use_pure_pip
        and not (
            sys.platform == "win32"
            and package_description == UV  # unable to overwrite itself
        )
        else [executable, "-m", PIP, "install", "-U"]
    )
    return call(args=[*args, package_description], creationflags=creationflags)


def update_packages() -> None:
    ap: argparse.ArgumentParser = argparse.ArgumentParser(
        description="update all Python packages in an environment",
        epilog="currently, `requirements.txt`, `pyproject.toml`, &c. are ignored",
    )
    ap.add_argument("--pre", action="store_true", help="include pre-release versions")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="check for updates, but don't perform actual updating",
    )
    ap.add_argument(
        "--pure-pip",
        action="store_true",
        help="use Python's pip, despite other tools might be available",
    )
    ap.add_argument(
        "venv",
        type=Path,
        nargs="?",
        default=Path(sys.exec_prefix),
        help="a path to a virtual environment to perform the update in (current one by default)",
    )
    ap.add_argument(
        "-s",
        action="store_true",
        help="do it safely, slowly, and single-threaded",
    )
    args: argparse.Namespace = ap.parse_intermixed_args()

    priority_packages: list[str] = [PIP, "setuptools", "wheel"]
    outdated_packages: dict[Distribution, str] = {}

    distribution: Distribution
    with ThreadPoolExecutor(max_workers=1 if args.s else 16) as executor:
        package_version_workers: dict[Future[Sequence[str]], Distribution] = {
            executor.submit(read_package_versions, distribution, pre=args.pre): (
                distribution
            )
            for distribution in list_packages(
                (args.venv, args.venv / ".venv", args.venv / "venv")
            )
        }
        future: Future[Sequence[str]]
        for future in as_completed(package_version_workers):
            distribution = package_version_workers[future]
            try:
                package_versions: Sequence[str] = future.result()
            except Exception as ex:
                print(
                    f"Failed to load available versions for {distribution.name}: {ex}",
                    file=sys.stderr,
                )
            else:
                if package_versions:
                    version: str
                    try:
                        version = distribution.origin.vcs_info.commit_id
                    except AttributeError:
                        version = distribution.version
                        if version != (
                            latest_version := package_versions[-1]
                        ) and compare_versions(version, latest_version, operator.gt):
                            print("WARNING:", end=" ")
                    if version != (latest_version := package_versions[-1]):
                        print(
                            f"{distribution.name} is {version}, "
                            f"however {latest_version} available"
                        )
                        outdated_packages[distribution] = latest_version

    if not outdated_packages:
        print("No packages to update")
        return

    executable: str | None = (
        which(
            "python", path=args.venv / ("Scripts" if sys.platform == "win32" else "bin")
        )
        or which(
            "python",
            path=args.venv
            / ".venv"
            / ("Scripts" if sys.platform == "win32" else "bin"),
        )
        or which(
            "python",
            path=args.venv / "venv" / ("Scripts" if sys.platform == "win32" else "bin"),
        )
        or which("python", path=args.venv)
        or which("python", path=args.venv / ".venv")
        or which("python", path=args.venv / "venv")
    )

    if not executable:
        print(f"Python environment is not found at {args.venv}")
        return

    if args.dry_run:
        return

    for pp in priority_packages:
        for op, latest_version in outdated_packages.items():
            if pp == op.name:
                print(f"Updating {pp}")
                ret = update_package(
                    op,
                    executable=executable,
                    use_pure_pip=args.pure_pip,
                    latest_version=latest_version,
                )
                if ret:
                    return
                break

    for op, latest_version in outdated_packages.items():
        if op.name is None:
            # already updated
            continue
        if op.name in ("pip-updater" or "pip_updater"):
            continue
        print(f"Updating {op.name}")
        ret = update_package(
            op,
            executable=executable,
            use_pure_pip=args.pure_pip,
            latest_version=latest_version,
        )
        if ret:
            # continue with other packages
            pass

    for op, latest_version in outdated_packages.items():
        if op.name is None:
            # already updated
            continue
        print(f"Updating {op.name}")
        update_package(
            op,
            executable=executable,
            use_pure_pip=args.pure_pip,
            latest_version=latest_version,
            creationflags=DETACHED_PROCESS,
        )
        exit()


def site_paths(prefixes: Iterable[str | os.PathLike[str]] | None = None) -> list[str]:
    if prefixes is not None:
        prefixes = set(map(str, prefixes))

    if (
        cached_result := next(
            (
                _paths
                for _prefixes, _paths in getattr(site_paths, "last_calls", [])
                if _prefixes == prefixes
            ),
            None,
        )
    ) is not None:
        return cached_result

    paths: set[Path] = set()
    path: Path
    for path in map(Path, site.getsitepackages(prefixes)):
        if not path.exists() or any(p.samefile(path) for p in paths):
            continue
        paths.add(path.resolve())

    result: list[str] = list(map(str, paths))

    setattr(
        site_paths,
        "last_calls",
        getattr(site_paths, "last_calls", []) + [(prefixes, result)],
    )

    return result


def list_packages(
    prefixes: Iterable[str | os.PathLike[str]] | None = None,
) -> Iterator[Distribution]:
    distribution: Distribution
    for distribution in Distribution.discover(
        context=DistributionFinder.Context(path=site_paths(prefixes))
    ):
        package_name: str = distribution.name
        if (
            installer_file := next(
                # `Path(str(f))` helps circumvent the mixture of separators on Windows
                (f for f in distribution.files if Path(str(f)).name == "INSTALLER"),
                None,
            )
        ) is None:
            print(f"Unknown installer for {package_name}", file=sys.stderr)
            continue
        elif (installer := installer_file.read_text(encoding="utf-8").strip()) not in (
            PIP,
            UV,
        ):
            print(f"{package_name} installed with {installer}", file=sys.stderr)
            continue
        yield distribution


class VersionInfoType(Protocol):
    @property
    def major(self) -> int:
        return 0

    @property
    def minor(self) -> int:
        return 0

    @property
    def micro(self) -> int:
        return 0

    @property
    def releaselevel(self) -> Literal["alpha", "beta", "candidate", "final"]:
        return "final"

    @property
    def serial(self) -> int:
        return 0


def list_packages_tree() -> Graph:
    graph: Graph = Graph()
    pattern: re.Pattern[str] = re.compile(
        r"^(?P<package>[\w\-]*)[^;]*(;\s*\[(?P<extra>[^]]+)])?$"
    )
    extras: dict[str, set[str]] = {}
    packages: set[str] = set()

    def format_full_version(info: VersionInfoType) -> str:
        version: str = f"{info.major}.{info.minor}.{info.micro}"
        kind: Literal["alpha", "beta", "candidate", "final"] = info.releaselevel
        if kind != "final":
            version += kind[0] + str(info.serial)
        return version

    distribution: Distribution
    for distribution in Distribution.discover():
        package_name: str = distribution.name
        if distribution.origin is not None:
            try:
                print(
                    f"{package_name} installed directly from {distribution.origin.url}",
                    file=sys.stderr,
                )
            except AttributeError:
                print(f"{package_name} installed directly from a URL", file=sys.stderr)
        if (
            installer_file := next(
                (f for f in distribution.files if f.name == "INSTALLER"), None
            )
        ) is None:
            print(f"Unknown installer for {package_name}", file=sys.stderr)
            continue
        elif (installer := installer_file.read_text(encoding="utf-8").strip()) not in (
            PIP,
            UV,
        ):
            print(f"{package_name} installed with {installer}", file=sys.stderr)
            continue
        packages.add(package_name)

        for line in distribution.requires or []:
            if ";" in line:
                constrains: str
                line, constrains = line.split(";", maxsplit=1)
                if not eval(
                    constrains.strip(),
                    dict(
                        os_name=os.name,
                        sys_platform=sys.platform,
                        platform_machine=platform.machine(),
                        platform_python_implementation=platform.python_implementation(),
                        platform_release=platform.release(),
                        platform_system=platform.system(),
                        platform_version=platform.version(),
                        python_version=".".join(platform.python_version_tuple()[:2]),
                        python_full_version=platform.python_version(),
                        implementation_name=sys.implementation.name,
                        implementation_version=(
                            format_full_version(sys.implementation.version)
                            if hasattr(sys, "implementation")
                            else "0"
                        ),
                        extra=extras.get(package_name, ""),
                    ),
                ):
                    continue
            match: re.Match[str] | None = pattern.match(line)
            if match is not None:
                graph.add_node(package_name, package := match.group("package"))
                if (extra := match.group("extra")) is not None:
                    extras.get(package, set()).update(
                        set(
                            e_match.group()
                            for e in extra.strip(",")
                            if (e_match := pattern.match(e.strip())) is not None
                        )
                    )

    for package_name in packages:
        if not graph.find(package_name):
            graph.add_node(package_name)
    return graph


def orphaned_packages() -> frozenset[str]:
    return list_packages_tree().top() - frozenset((PIP, "setuptools"))


def print_orphaned_packages() -> None:
    _orphaned_packages: frozenset[str] = orphaned_packages()
    if _orphaned_packages:
        print("The following packages are not required by other packages:")
        package_name: str
        for package_name in _orphaned_packages:
            print(f"    • {package_name}")
    else:
        print("All packages are required.")
