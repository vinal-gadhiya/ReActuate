"""Defer optional RADIO imports until a RADIO backbone is actually requested."""

import argparse
import ast
from pathlib import Path


def defer_radio_import(path, symbol):
    source = path.read_text()
    tree = ast.parse(source)
    imports = [
        node for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == 'radio_model'
        and node.level == 1
    ]
    if len(imports) != 1 or [(item.name, item.asname) for item in imports[0].names] != [(symbol, None)]:
        raise ValueError(f'Unexpected RADIO import in {path}')
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == symbol
    ]
    if len(calls) != 1:
        raise ValueError(f'Expected one {symbol} constructor in {path}')
    statement = calls[0]
    while not isinstance(statement, ast.stmt):
        statement = parents[statement]
    if statement.col_offset == 0:
        raise ValueError(f'{symbol} must only be constructed inside a branch')
    # Preserve source comments/licenses; only relocate the existing import.
    lines = source.splitlines(keepends=True)
    original = imports[0]
    relocated = ' ' * statement.col_offset + f'from .radio_model import {symbol}\n'
    for index in range(len(lines), 0, -1):
        if index == statement.lineno:
            lines.insert(index - 1, relocated)
        if original.lineno <= index <= original.end_lineno:
            del lines[index - 1]
    patched = ''.join(lines)
    ast.parse(patched)
    path.write_text(patched)


def patch(root):
    directory = root / 'gr00t/model/backbone/eagle2_hg_model'
    defer_radio_import(directory / 'configuration_eagle2_5_vl.py', 'RADIOConfig')
    defer_radio_import(directory / 'modeling_eagle2_5_vl.py', 'RADIOModel')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    patch(parser.parse_args().root)
