"""Repair the cuSPARSELt 0.8.0 ARM64 wheel's incorrect internal SBSA tag."""

import base64
import csv
import ctypes
import hashlib
from importlib.metadata import distribution
from io import StringIO
import json
from pathlib import Path
import platform

OLD_TAG = 'Tag: py3-none-manylinux2014_sbsa'
NEW_TAG = 'Tag: py3-none-manylinux2014_aarch64'


def digest(path):
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').digest()
    return 'sha256=' + base64.urlsafe_b64encode(value).decode().rstrip('=')


def repair(dist):
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        raise RuntimeError('This repair applies only to Linux aarch64')
    if dist.version != '0.8.0':
        raise RuntimeError(f'Unreviewed cuSPARSELt version: {dist.version}')
    files = list(dist.files or [])
    wheel_entry = next(item for item in files if str(item).endswith('.dist-info/WHEEL'))
    record_entry = next(item for item in files if str(item).endswith('.dist-info/RECORD'))
    library_entry = next(item for item in files if str(item).endswith('/lib/libcusparseLt.so.0'))
    wheel = Path(dist.locate_file(wheel_entry))
    record = Path(dist.locate_file(record_entry))
    library = Path(dist.locate_file(library_entry))
    text = wheel.read_text()
    tags = [line for line in text.splitlines() if line.startswith('Tag:')]
    if tags == [NEW_TAG]:
        return {'status': 'already-correct', 'version': dist.version}
    if tags != [OLD_TAG]:
        raise RuntimeError(f'Unexpected cuSPARSELt wheel tags: {tags}')

    with library.open('rb') as stream:
        header = stream.read(20)
    if len(header) != 20 or header[:6] != b'\x7fELF\x02\x01' or int.from_bytes(header[18:20], 'little') != 183:
        raise RuntimeError('cuSPARSELt library is not a little-endian ELF64 AArch64 binary')

    rows = list(csv.reader(StringIO(record.read_text())))
    by_path = {row[0]: row for row in rows}
    for entry, path in ((library_entry, library), (wheel_entry, wheel)):
        row = by_path[str(entry)]
        if row[1] != digest(path) or int(row[2]) != path.stat().st_size:
            raise RuntimeError(f'Installed file differs from its RECORD: {entry}')
    ctypes.CDLL(str(library))  # Loads the binary without requiring a GPU kernel.

    # Change metadata only, preserving the pinned binary and package version.
    wheel.write_text(text.replace(OLD_TAG, NEW_TAG))
    row = by_path[str(wheel_entry)]
    row[1:] = [digest(wheel), str(wheel.stat().st_size)]
    buffer = StringIO(newline='')
    csv.writer(buffer).writerows(rows)
    record.write_text(buffer.getvalue())
    return {
        'status': 'repaired', 'package': 'nvidia-cusparselt-cu13',
        'version': dist.version, 'old_tag': OLD_TAG, 'new_tag': NEW_TAG,
        'library_sha256': digest(library),
    }


if __name__ == '__main__':
    result = repair(distribution('nvidia-cusparselt-cu13'))
    print(json.dumps(result, indent=2))
    Path('/opt/build-info/cusparselt-metadata.json').write_text(json.dumps(result, indent=2) + '\n')
