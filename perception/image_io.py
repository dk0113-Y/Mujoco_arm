from __future__ import annotations

from pathlib import Path
import struct
import zlib

import numpy as np


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(name: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + name
        + data
        + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
    )


def _write_png(
    path: str | Path,
    array: np.ndarray,
    *,
    bit_depth: int,
    color_type: int,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = array.shape[:2]
    rows = b"".join(b"\x00" + array[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    output_path.write_bytes(
        PNG_SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(rows, level=6))
        + _chunk(b"IEND", b"")
    )


def save_rgb_png(path: str | Path, rgb: np.ndarray) -> None:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("RGB PNG input must have shape (H, W, 3) and dtype uint8")
    _write_png(path, np.ascontiguousarray(image), bit_depth=8, color_type=2)


def save_mask_png(path: str | Path, mask: np.ndarray) -> None:
    image = np.asarray(mask, dtype=np.uint8) * 255
    if image.ndim != 2:
        raise ValueError("Mask PNG input must be a 2D array")
    _write_png(path, np.ascontiguousarray(image), bit_depth=8, color_type=0)


def save_depth_millimeters_png(path: str | Path, depth: np.ndarray) -> None:
    values = np.asarray(depth, dtype=float)
    if values.ndim != 2:
        raise ValueError("Depth PNG input must be a 2D array")
    valid = np.isfinite(values) & (values > 0.0) & (values < 65.535)
    millimeters = np.zeros(values.shape, dtype=np.uint16)
    millimeters[valid] = np.rint(values[valid] * 1000.0).astype(np.uint16)
    # PNG stores 16-bit samples in network byte order.
    _write_png(
        path,
        np.ascontiguousarray(millimeters.astype(">u2")),
        bit_depth=16,
        color_type=0,
    )


def save_depth_preview_png(path: str | Path, depth: np.ndarray) -> None:
    values = np.asarray(depth, dtype=float)
    if values.ndim != 2:
        raise ValueError("Depth preview input must be a 2D array")
    valid = np.isfinite(values) & (values > 0.0)
    image = np.zeros(values.shape, dtype=np.uint8)
    if np.any(valid):
        near = float(np.min(values[valid]))
        far = float(np.max(values[valid]))
        if far > near:
            # Brighter means closer; this file is visualization only.
            image[valid] = np.rint(
                255.0 * (far - values[valid]) / (far - near)
            ).astype(np.uint8)
        else:
            image[valid] = 255
    _write_png(path, np.ascontiguousarray(image), bit_depth=8, color_type=0)
