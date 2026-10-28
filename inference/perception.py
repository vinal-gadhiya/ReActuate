"""Bounded video encoding and SAM clip-local tracking records."""

import base64
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


def video_part(frames, fps=5):
    if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3
            or not 2 <= len(frames) <= 8 or max(frames.shape[1:3]) > 1024):
        raise ValueError('Expected 2–8 bounded RGB video samples')
    height, width = frames.shape[1:3]
    if height % 2 or width % 2:
        raise ValueError('Video samples must have even dimensions to preserve SAM coordinates')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'clip.mp4'
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
            '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(path),
        ], input=frames.tobytes(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            check=True, timeout=30)
        encoded = base64.b64encode(path.read_bytes()).decode()
    return {'type': 'input_video', 'video_url': 'data:video/mp4;base64,' + encoded}


def tracking_records(text, width, height, frame_count):
    records = []
    chunks = re.split(r'<(\d+)f>', text)
    for index in range(1, len(chunks) - 1, 2):
        frame = int(chunks[index])
        if not 0 <= frame < frame_count:
            continue
        for match in re.finditer(r'(\d+)<\|box;([^|]+)\|>', chunks[index + 1]):
            fields = dict(part.split('=', 1) for part in match[2].split(';') if '=' in part)
            try:
                box = {key: int(fields[key]) for key in ('x1', 'y1', 'x2', 'y2', 'w', 'h')}
            except (ValueError, KeyError):
                continue
            if (box['w'] == width and box['h'] == height and
                    0 <= box['x1'] <= box['x2'] < width and 0 <= box['y1'] <= box['y2'] < height):
                records.append({'frame': frame, 'object_id': int(match[1]), **box})
    return sorted(records, key=lambda item: (item['frame'], item['object_id']))[:128]
