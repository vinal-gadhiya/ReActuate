"""Container-only regression tests for optional RADIO imports."""

import ast
from pathlib import Path
import tempfile
import unittest

from inference.patch_radio_imports import defer_radio_import


class RadioImportTests(unittest.TestCase):
    def test_siglip_path_does_not_import_radio(self):
        for symbol in ('RADIOConfig', 'RADIOModel'):
            with self.subTest(symbol=symbol), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'example.py'
                path.write_text(
                    '# license remains\n'
                    f'from .radio_model import {symbol}\n'
                    'def create(model_type):\n'
                    '    if model_type == "radio":\n'
                    f'        return {symbol}()\n'
                    '    return "siglip"\n'
                )
                defer_radio_import(path, symbol)
                patched = path.read_text()
                self.assertTrue(patched.startswith('# license remains\n'))
                namespace = {'__name__': 'example'}
                exec(compile(patched, str(path), 'exec'), namespace)
                self.assertEqual(namespace['create']('siglip'), 'siglip')
                tree = ast.parse(patched)
                self.assertFalse(any(isinstance(node, ast.ImportFrom) for node in tree.body))
                # Selecting RADIO still requires its real optional dependency.
                with self.assertRaises(ImportError):
                    namespace['create']('radio')

    def test_unexpected_source_fails_instead_of_silently_skipping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'example.py'
            path.write_text('def create():\n    return "siglip"\n')
            with self.assertRaises(ValueError):
                defer_radio_import(path, 'RADIOConfig')
