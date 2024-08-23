# coding: utf-8
import json
import os
import platform
import re
import site
import sys
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from functools import cache
from html.parser import HTMLParser
from http.client import HTTPResponse
from pathlib import Path
from shutil import which
from subprocess import PIPE, Popen
from typing import (
    Any,
    Final,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    Protocol,
    Sequence,
)
from urllib.error import HTTPError
from urllib.request import urlopen

__all__ = [
    "Graph",
    "read_package_versions",
    "list_packages_tree",
    "update_packages",
    "update_package",
    "orphaned_packages",
    "print_orphaned_packages",
]

DIST_INFO_PATTERN: Final[str] = "*.dist-info"

PIP: Final[str] = "pip"
UV: Final[str] = "uv"
UV_CMD: Final[str | None] = which(UV)

VCS_VERSION_TEMPLATES: Final[dict[str, Sequence[str]]] = {
    "git": ["git", "ls-remote", "--heads", "{url}"]
}


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


def parse_table(table_text: str) -> list[dict[str, str]]:
    if not table_text:
        return []

    text_lines: list[str] = table_text.splitlines()
    rules: list[str] = [line for line in text_lines if set(line) == set("- ")]
    if len(rules) != 1:
        raise RuntimeError("Failed to parse the table")
    if text_lines.index(rules[0]) != 1:
        raise RuntimeError("Failed to parse the table")
    cols: list[int] = [len(rule) for rule in rules[0].split()]
    titles: list[str] = []
    offset: int = 0
    for col in cols:
        titles.append(text_lines[0][offset : (offset + col)].strip())
        offset += col + 1
    data: list[dict[str, str]] = []
    for line_no in range(2, len(text_lines)):
        data.append(dict())
        offset = 0
        for col, title in zip(cols, titles):
            data[-1][title] = text_lines[line_no][offset : (offset + col)].strip()
            offset += col + 1
    return data


def find_line(lines: list[str], prefix: str, remove_prefix: bool = True) -> str:
    line: str = ""
    for line in lines:
        if line.startswith(prefix):
            if remove_prefix:
                return line.removeprefix(prefix)
            return line
    return line


def find_lines(
    lines: list[str], prefix: str, remove_prefix: bool = True
) -> Iterator[str]:
    line: str
    for line in lines:
        if line.startswith(prefix):
            if remove_prefix:
                yield line.removeprefix(prefix)
            else:
                yield line


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
            if self._yanked == "Not ready":
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


class PackageData(NamedTuple):
    name: str
    version: str
    aux_data: dict[str, Any] = {}


def read_package_versions(
    package_data: PackageData,
    pre: bool = False,
) -> Sequence[str]:
    def read_package_versions_pip() -> Sequence[str]:
        r: HTTPResponse
        parser: PackageFileParser = PackageFileParser(package_data.name, pre=pre)
        try:
            with urlopen(
                f"https://pypi.org/simple/{package_data.name.replace('_', '-')}/"
            ) as r:
                parser.feed(r.read().decode())
        except HTTPError as ex:
            if ex.code == 404:
                print(f"{package_data.name} not found", file=sys.stderr)
            else:
                print(f"{package_data.name}: {ex!s}", file=sys.stderr)
        return parser.versions

    def read_package_versions_vcs() -> Sequence[str]:
        p: Popen[bytes]
        with Popen(
            args=[arg.format(url=url) for arg in VCS_VERSION_TEMPLATES[vcs]],
            stdout=PIPE,
        ) as p:
            return [p.stdout.read().split()[0].decode()]

    if package_data.aux_data:
        url: str = package_data.aux_data.get("url", "")
        vcs: str = package_data.aux_data.get("vcs_info", {}).get("vcs", "")
        if url and vcs in VCS_VERSION_TEMPLATES:
            return read_package_versions_vcs()

    return read_package_versions_pip()


def update_package(package_data: PackageData) -> int:
    package_description: str = package_data.name
    if package_data.aux_data:
        url: str = package_data.aux_data.get("url", "")
        vcs: str = package_data.aux_data.get("vcs_info", {}).get("vcs", "")
        package_description = "+".join((vcs, url))

    executable: list[str] = (
        [UV_CMD, PIP] if UV_CMD is not None else [sys.executable, "-m", PIP]
    )
    p: Popen[bytes]
    with Popen(
        args=[*executable, "install", "-U", package_description],
    ) as p:
        return p.returncode


def update_packages() -> None:
    pre: bool = "--pre" in sys.argv
    dry_run: bool = "--dry-run" in sys.argv
    priority_packages: list[str] = [PIP, "setuptools", "wheel"]
    outdated_packages: list[PackageData] = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        package_version_workers: dict[Future[Sequence[str]], PackageData] = {
            executor.submit(read_package_versions, package_data, pre=pre): (
                package_data
            )
            for package_data in list_packages()
        }
        future: Future[Sequence[str]]
        for future in as_completed(package_version_workers):
            package_data: PackageData = package_version_workers[future]
            try:
                package_versions: Sequence[str] = future.result()
            except Exception as ex:
                print(
                    f"Failed to load available versions for {package_data.name}: {ex}",
                    file=sys.stderr,
                )
            else:
                if package_versions and package_data.version != (
                    latest_version := package_versions[-1]
                ):
                    print(
                        f"{package_data.name} is {package_data.version}, "
                        f"however {latest_version} available"
                    )
                    outdated_packages.append(
                        PackageData(
                            package_data.name, latest_version, package_data.aux_data
                        )
                    )
    if not outdated_packages:
        print("No packages to update")
        return

    if dry_run:
        return

    for pp in priority_packages:
        for op in outdated_packages:
            if pp == op.name:
                print(f"Updating {pp}")
                ret = update_package(op)
                if ret:
                    return
                outdated_packages.remove(op)
                break
    for op in outdated_packages:
        package_path: Path | None = None
        for site_path in site_paths():
            dist_info_paths = [
                *site_path.glob(op.name + DIST_INFO_PATTERN),
                *site_path.glob(op.name.replace("-", "_") + DIST_INFO_PATTERN),
            ]
            if dist_info_paths:
                package_path = dist_info_paths[-1]
                break
        if package_path is not None and package_path.exists():
            pd: PackageData = read_package_data(package_path)
            if pd.version == op.version:
                print(f"Already updated {op.name}")
                continue
        print(f"Updating {op.name}")
        ret = update_package(op)
        if ret:
            # continue with other packages
            pass


@cache
def site_paths() -> frozenset[Path]:
    paths: set[Path] = set()
    path: Path
    for path in map(Path, site.getsitepackages()):
        if not path.exists() or any(p.samefile(path) for p in paths):
            continue
        paths.add(path)
    return frozenset(paths)


def read_metadata(package_path: Path) -> list[str]:
    return (package_path / "METADATA").read_text(encoding="utf-8").splitlines()


def read_package_data(package_path: Path) -> PackageData:
    metadata: list[str] = read_metadata(package_path)
    package_name: str = find_line(metadata, "Name: ")
    package_version: str = find_line(metadata, "Version: ")
    direct_url_data: dict[str, Any] = {}
    if (package_path / "direct_url.json").exists():
        print(f"{package_name} installed directly from a URL", file=sys.stderr)
        direct_url_data = json.loads((package_path / "direct_url.json").read_bytes())
        package_version = direct_url_data.get("vcs_info", {}).get(
            "commit_id", package_version
        )
    return PackageData(package_name, package_version, direct_url_data)


def list_packages() -> Iterator[PackageData]:
    site_path: Path
    for site_path in site_paths():
        package_path: Path
        for package_path in site_path.glob(DIST_INFO_PATTERN):
            if package_path.name.startswith("~"):
                continue
            pd: PackageData = read_package_data(package_path)
            if not (installer_file := (package_path / "INSTALLER")).exists():
                print(f"Unknown installer for {pd.name}", file=sys.stderr)
                continue
            elif (
                installer := installer_file.read_text(encoding="utf-8").strip()
            ) not in (PIP, UV):
                print(f"{pd.name} installed with {installer}", file=sys.stderr)
                continue
            yield pd


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

    site_path: Path
    for site_path in site_paths():
        package_path: Path
        for package_path in site_path.glob(DIST_INFO_PATTERN):
            if package_path.name.startswith("~"):
                continue
            metadata: list[str] = read_metadata(package_path)
            package_name: str = find_line(metadata, "Name: ")
            if (package_path / "direct_url.json").exists():
                print(f"{package_name} installed directly from a URL", file=sys.stderr)
                continue
            if not (installer_file := (package_path / "INSTALLER")).exists():
                print(f"Unknown installer for {package_name}", file=sys.stderr)
                continue
            elif (
                installer := installer_file.read_text(encoding="utf-8").strip()
            ) not in (PIP, UV):
                print(f"{package_name} installed with {installer}", file=sys.stderr)
                continue
            packages.add(package_name)
            for line in find_lines(metadata, "Requires-Dist: "):
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
                            python_version=".".join(
                                platform.python_version_tuple()[:2]
                            ),
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
