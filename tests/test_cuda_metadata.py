"""Container-only tests for the narrowly scoped CUDA wheel metadata repair."""

import csv
from io import StringIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from inference.repair_cuda_metadata import NEW_TAG, OLD_TAG, digest, repair


class MetadataTests(unittest.TestCase):
    def test_repair_preserves_binary_and_updates_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = Path('package.dist-info/WHEEL')
            record = Path('package.dist-info/RECORD')
            library = Path('nvidia/cusparselt/lib/libcusparseLt.so.0')
            for entry in (wheel, record, library):
                (root / entry).parent.mkdir(parents=True, exist_ok=True)
            original = b'\x7fELF\x02\x01' + bytes(12) + (183).to_bytes(2, 'little')
            (root / library).write_bytes(original)
            (root / wheel).write_text('Wheel-Version: 1.0\n' + OLD_TAG + '\n')
            buffer = StringIO(newline='')
            csv.writer(buffer).writerows([
                [str(entry), digest(root / entry), (root / entry).stat().st_size]
                for entry in (wheel, library)
            ] + [[str(record), '', '']])
            (root / record).write_text(buffer.getvalue())
            dist = SimpleNamespace(version='0.8.0', files=[wheel, record, library], locate_file=lambda item: root / item)
            with patch('inference.repair_cuda_metadata.platform.system', return_value='Linux'), \
                 patch('inference.repair_cuda_metadata.platform.machine', return_value='aarch64'), \
                 patch('inference.repair_cuda_metadata.ctypes.CDLL') as load:
                self.assertEqual(repair(dist)['status'], 'repaired')
                load.assert_called_once()
                self.assertEqual(repair(dist)['status'], 'already-correct')
            self.assertEqual((root / library).read_bytes(), original)
            self.assertIn(NEW_TAG, (root / wheel).read_text())
            rows = {row[0]: row for row in csv.reader(StringIO((root / record).read_text()))}
            self.assertEqual(rows[str(wheel)][1], digest(root / wheel))

    def test_reject_unreviewed_versions(self):
        with patch('inference.repair_cuda_metadata.platform.system', return_value='Linux'), \
             patch('inference.repair_cuda_metadata.platform.machine', return_value='aarch64'):
            with self.assertRaisesRegex(RuntimeError, 'Unreviewed'):
                repair(SimpleNamespace(version='0.9.0'))
