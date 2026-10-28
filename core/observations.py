"""Public RGB integrity checks shared by simulator and supervisor."""

import hashlib
import json

import numpy as np


def array_digest(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str((value.shape, str(value.dtype))).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def rgb_diagnostics(images):
    result = {}
    for key, value in images.items():
        value = np.asarray(value)
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f'Invalid RGB frame: {key}')
        pixels = value.astype(np.float32)
        horizontal = float(np.mean(np.abs(np.diff(pixels, axis=1))))
        vertical = float(np.mean(np.abs(np.diff(pixels, axis=0))))
        std = float(pixels.std())
        result[key] = {'sha256': array_digest(value), 'shape': list(value.shape),
                       'neighbor_mean_abs_diff': (horizontal + vertical) / 2, 'std': std,
                       'suspected_static_noise': horizontal > 55 and vertical > 55 and std > 50}
    return result


def check_rgb(images, directory=None, step=0):
    diagnostics = rgb_diagnostics(images)
    if diagnostics and all(item['suspected_static_noise'] for item in diagnostics.values()):
        if directory is not None:
            from PIL import Image
            np.savez_compressed(directory / f'suspected-noise-step-{step}.npz', **images)
            for number, frame in enumerate(images.values()):
                Image.fromarray(frame).save(directory / f'suspected-noise-step-{step}-camera-{number}.png')
            (directory / 'image_quality_error.json').write_text(json.dumps({
                'step': step, 'cameras': diagnostics,
                'note': 'Heuristic suspected corruption; inspect the saved pixels before diagnosing the renderer.',
            }, indent=2) + '\n')
        raise RuntimeError('All current RGB cameras look like static noise; saved raw frames when an artifact directory is available')
    return diagnostics
