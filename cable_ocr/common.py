import hashlib
import json
import os
import tempfile
from pathlib import Path


class PhaseError(Exception):
    """A safe, typed error. Never include request URLs/tokens or secrets."""

    def __init__(self, code, details=None):
        self.code = code
        self.details = details or {}
        super().__init__(code)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(value) -> str:
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())


def save_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_json(path, value):
    save_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode())


def box_area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])


def intersection(a, b):
    return box_area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def contains_center(container, fragment):
    x, y = (fragment[0]+fragment[2])/2, (fragment[1]+fragment[3])/2
    return container[0] <= x < container[2] and container[1] <= y < container[3]


def bbox_union(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
