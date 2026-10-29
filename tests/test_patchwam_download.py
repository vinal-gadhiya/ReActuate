"""Exercise download pinning/integrity with fake HF responses, no model downloads."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.patchwam import CHECKPOINT_REF
from inference import download_patchwam


class PatchWAMDownloadTests(unittest.TestCase):
    def test_resolved_revisions_are_reused_and_files_are_hashed(self):
        requests = []
        checkpoint_bytes = b'fake checkpoint for transport tests'
        expected = hashlib.sha256(checkpoint_bytes).hexdigest()
        resolved = {'revision': 'a' * 40}

        class API:
            def model_info(self, repo, revision):
                requests.append((repo, revision))
                return SimpleNamespace(sha=CHECKPOINT_REF if repo == download_patchwam.CHECKPOINT_REPO
                                       else revision if revision != 'main' else resolved['revision'])

        def snapshot(repo, revision, local_dir, allow_patterns):
            destination = Path(local_dir)
            destination.mkdir(parents=True, exist_ok=True)
            if 'checkpoint' == destination.name:
                (destination / 'step_559660.pt').write_bytes(checkpoint_bytes)
            elif destination.name == 'qwen':
                (destination / 'config.json').write_text('{}')
                (destination / 'model.safetensors').write_bytes(b'fake qwen')
            else:
                (destination / allow_patterns[0]).write_bytes(b'fake base')

        fake_hub = SimpleNamespace(HfApi=API, snapshot_download=snapshot)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {'huggingface_hub': fake_hub}), \
                patch.object(download_patchwam, 'CHECKPOINT_SHA256', expected), \
                patch.object(sys, 'argv', ['download_patchwam', '--output', directory]):
            download_patchwam.main()
            manifest = json.loads((Path(directory) / 'manifest.json').read_text())
            self.assertEqual(manifest['checkpoint']['sha256'], expected)
            self.assertEqual(manifest['assets']['flux']['revision'], 'a' * 40)
            self.assertEqual(manifest['files']['qwen/model.safetensors'], hashlib.sha256(b'fake qwen').hexdigest())
            requests.clear()
            resolved['revision'] = 'b' * 40
            download_patchwam.main()
            self.assertTrue(all(revision != 'main' for _, revision in requests))
            self.assertEqual(json.loads((Path(directory) / 'manifest.json').read_text()), manifest)

    def test_corrupt_checkpoint_never_publishes_ready_manifest(self):
        class API:
            def model_info(self, repo, revision):
                return SimpleNamespace(sha=CHECKPOINT_REF if repo == download_patchwam.CHECKPOINT_REPO else 'a' * 40)

        def snapshot(repo, revision, local_dir, allow_patterns):
            destination = Path(local_dir)
            destination.mkdir(parents=True, exist_ok=True)
            name = 'model.safetensors' if destination.name == 'qwen' else allow_patterns[0]
            (destination / name).write_bytes(b'corrupt')

        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {'huggingface_hub': SimpleNamespace(HfApi=API, snapshot_download=snapshot)}), \
                patch.object(sys, 'argv', ['download_patchwam', '--output', directory]):
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                download_patchwam.main()
            self.assertFalse((Path(directory) / 'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
