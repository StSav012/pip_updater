# coding: utf-8
from __future__ import annotations

import os
import platform
import re
import site
import sys
from pathlib import Path
from subprocess import PIPE, Popen
from typing import Iterable, Iterator


class Graph:
    class Node:
        def __init__(self, value: str) -> None:
            self.value: str = value
            self.children: set[Graph.Node] = set()
            self.parents: set[Graph.Node] = set()

        def __repr__(self) -> str:
            return f'{self.value!r}: ' + '{' + ', '.join(repr(child) for child in self.children) + '}'

        def __str__(self) -> str:
            return self.value

        def __eq__(self, other: Graph.Node | str) -> bool:
            if isinstance(other, str):
                return self.value == other
            return self.value == other.value

        def __hash__(self) -> int:
            return hash(self.value)

        def __getitem__(self, item: str) -> Graph.Node:
            if self == item:
                return self
            node: Graph.Node
            for node in self.children:
                if node == item:
                    return node
            raise IndexError(f'Item {item} not found')

    def __init__(self) -> None:
        self.nodes: set[Graph.Node] = set()

    def find(self, value: str) -> Graph.Node | None:
        def recursive_find(where: Iterable[Graph.Node]) -> Graph.Node | None:
            node: Graph.Node
            for node in where:
                if node == value:
                    return node
                found: None | Graph.Node = recursive_find(node.children)
                if found is not None:
                    return found

        return recursive_find(self.nodes)

    def add_node(self, parent: str | Graph.Node, value: str | None = None) -> None:
        if isinstance(parent, str):
            found_parent: Graph.Node | None = self.find(parent)
            if found_parent is None:
                parent = Graph.Node(parent)
                self.nodes.add(parent)
            else:
                parent = found_parent
        if value is not None:
            node: Graph.Node = Graph.Node(value)
            node.parents.add(parent)
            parent.children.add(node)

    def __repr__(self) -> str:
        node: Graph.Node
        return self.__class__.__name__ + '(\n' + ',\n'.join(repr(node) for node in self.nodes) + ')'

    def __invert__(self) -> Graph:
        graph: Graph = Graph()

        def recursive_children(child_node: Graph.Node) -> None:
            def recursive_parents(parent_node: Graph.Node, parents: set[Graph.Node]) -> None:
                if not parents:
                    return
                parent: Graph.Node
                for parent in parents:
                    parent_node.children.add(Graph.Node(parent.value))
                parents: set[Graph.Node] = parent_node.children.copy()
                while parents:
                    parent = parents.pop()
                    recursive_parents(parent_node[parent.value], parent.parents)

            if not child_node.children:
                new_node: Graph.Node = Graph.Node(child_node.value)
                new_node.children = set(Graph.Node(parent.value) for parent in child_node.parents)
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
    text_lines: list[str] = table_text.splitlines()
    rules: list[str] = [line for line in text_lines if set(line) == set('- ')]
    if len(rules) != 1:
        raise RuntimeError('Failed to parse the table')
    if text_lines.index(rules[0]) != 1:
        raise RuntimeError('Failed to parse the table')
    cols: list[int] = [len(rule) for rule in rules[0].split()]
    titles: list[str] = []
    offset: int = 0
    for col in cols:
        titles.append(text_lines[0][offset:(offset + col)].strip())
        offset += col + 1
    data: list[dict[str, str]] = []
    for line_no in range(2, len(text_lines)):
        data.append(dict())
        offset = 0
        for col, title in zip(cols, titles):
            data[-1][title] = text_lines[line_no][offset:(offset + col)].strip()
            offset += col + 1
    return data


def find_line(lines: list[str], prefix: str, remove_prefix: bool = True) -> str:
    line: str = ''
    for line in lines:
        if line.startswith(prefix):
            if remove_prefix:
                return line.removeprefix(prefix)
            return line
    return line


def find_lines(lines: list[str], prefix: str, remove_prefix: bool = True) -> Iterator[str]:
    line: str
    for line in lines:
        if line.startswith(prefix):
            if remove_prefix:
                yield line.removeprefix(prefix)
            else:
                yield line


def update_package(package_name: str) -> tuple[str, str, int]:
    p: Popen
    with Popen(args=[sys.executable, '-m', 'pip', 'install', '-U', package_name],
               stdout=PIPE, stderr=PIPE, text=True) as p:
        err: str = p.stderr.read()
        if err:
            sys.stderr.write(err)
        return p.stdout.read(), err, p.returncode


def update_packages() -> None:
    priority_packages: list[str] = ['pip', 'setuptools', 'wheel']
    err: str
    p: Popen
    with Popen(args=[sys.executable, '-m', 'pip', 'list', '--outdated'], stdout=PIPE, stderr=PIPE, text=True) as p:
        err = p.stderr.read()
        if p.returncode:
            sys.stderr.write(err)
            return
        outdated_packages: list[str] = [item['Package'] for item in parse_table(p.stdout.read())]
    for pp in priority_packages:
        if pp in outdated_packages:
            print(f'Updating {pp}')
            out, err, ret = update_package(pp)
            print(out)
            if ret:
                sys.stderr.write(err)
                return
            outdated_packages.remove(pp)
    for op in outdated_packages:
        print(f'Updating {op}')
        out, err, ret = update_package(op)
        print(out)
        if ret:
            sys.stderr.write(err)
            # continue with other packages


def list_packages() -> Iterator[tuple[str, str]]:
    site_paths: set[Path] = set()
    site_path: Path
    for site_path in map(Path, site.getsitepackages()):
        if any(p.samefile(site_path) for p in site_paths):
            continue
        site_paths.add(site_path)

        package_path: Path
        for package_path in site_path.glob('*.dist-info'):
            package_name: str = package_path.name.removesuffix('.dist-info')
            if '-' in package_name:
                yield tuple(package_name.split('-', maxsplit=1))
            else:
                yield package_name, ''


def list_packages_tree() -> Graph:
    site_paths: set[Path] = set()
    graph: Graph = Graph()
    pattern: re.Pattern[str] = re.compile(r'^(?P<package>[\w\-]*)\s*(\[(?P<extra>[^]]+)])?')
    extras: dict[str, set[str]] = {}

    def format_full_version(info) -> str:
        version: str = f'{info.major}.{info.minor}.{info.micro}'
        kind: str = info.releaselevel
        if kind != 'final':
            version += kind[0] + str(info.serial)
        return version

    site_path: Path
    for site_path in map(Path, site.getsitepackages()):
        if any(p.samefile(site_path) for p in site_paths):
            continue
        site_paths.add(site_path)

        package_path: Path
        for package_path in site_path.glob('*.dist-info'):
            metadata: list[str] = (package_path / 'METADATA').read_text().splitlines()
            package_name: str = find_line(metadata, 'Name: ')
            for line in find_lines(metadata, 'Requires-Dist: '):
                if ';' in line:
                    constrains: str
                    line, constrains = line.split(';', maxsplit=1)
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
                                python_version='.'.join(platform.python_version_tuple()[:2]),
                                python_full_version=platform.python_version(),
                                implementation_name=sys.implementation.name,
                                implementation_version=(format_full_version(sys.implementation.version)
                                if hasattr(sys, 'implementation') else '0'),
                                extra=extras.get(package_name, ''),
                            ),
                    ):
                        continue
                match: re.Match[str] | None = pattern.search(line)
                if match is not None:
                    graph.add_node(package_name, package := match.group('package'))
                    if (extra := match.group('extra')) is not None:
                        extras.get(package, set()).update(set(pattern.search(e.strip()).group()
                                                              for e in extra.strip(',')))

            graph.add_node(package_name)
    return graph


def orphaned_packages() -> frozenset[str]:
    return list_packages_tree().top() - frozenset(('pip', 'setuptools'))


def print_orphaned_packages() -> None:
    _orphaned_packages: frozenset[str] = orphaned_packages()
    if _orphaned_packages:
        print('The following packages are not required by other packages:')
        package_name: str
        for package_name in _orphaned_packages:
            print(f'    • {package_name}')
    else:
        print('All packages are required.')
