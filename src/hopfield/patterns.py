"""Letter patterns: loading, noise, classification of recalled states."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_letters(
    path: str | Path, expected_shape: tuple[int, int] | None = None
) -> tuple[dict[str, np.ndarray], tuple[int, int]]:
    """Load letters from a JSON file and convert 0/1 to -1/+1.

    Returns a dict ``{name: 2D ±1 matrix}`` and the pattern shape.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    shape = tuple(raw.get("shape", (5, 5)))
    if expected_shape is not None and shape != tuple(expected_shape):
        raise ValueError(
            f"shape mismatch: file has {shape}, config expected {expected_shape}"
        )

    letters: dict[str, np.ndarray] = {}
    for name, matrix in raw["letters"].items():
        arr = np.asarray(matrix, dtype=int)
        if arr.shape != shape:
            raise ValueError(
                f"letter '{name}' has shape {arr.shape}, expected {shape}"
            )
        # map 0 -> -1, anything positive -> +1
        bipolar = np.where(arr > 0, 1, -1).astype(np.int8)
        letters[name] = bipolar
    return letters, shape


def flatten(pattern: np.ndarray) -> np.ndarray:
    return np.asarray(pattern, dtype=float).reshape(-1)


def unflatten(vec: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(vec).reshape(shape)


def add_noise(
    pattern: np.ndarray,
    n_flips: int | None = None,
    p: float | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Flip bits of ``pattern``. Either ``n_flips`` exact bits or ``p`` per-bit prob."""
    rng = rng if rng is not None else np.random.default_rng()
    flat = flatten(pattern).copy()
    n = flat.size

    if n_flips is not None and p is not None:
        raise ValueError("use either n_flips or p, not both")
    if n_flips is None and p is None:
        return flat

    if n_flips is not None:
        n_flips = int(n_flips)
        n_flips = max(0, min(n, n_flips))
        idx = rng.choice(n, size=n_flips, replace=False)
        flat[idx] *= -1
    else:
        mask = rng.random(n) < float(p)
        flat[mask] *= -1
    return flat


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Number of differing entries between two ±1 vectors."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return int(np.sum(a != b))


def match_stored(
    state: np.ndarray,
    stored: dict[str, np.ndarray],
    include_negative: bool = True,
) -> tuple[str, str] | tuple[None, str]:
    """Identify which stored pattern (if any) matches ``state``.

    Returns a tuple ``(name, kind)`` where ``kind`` is one of:
      - ``"pattern"`` -> identical to a stored letter
      - ``"negative"`` -> identical to the negative of a stored letter
      - ``"spurious"`` -> doesn't match any stored letter (nor negatives)
    """
    flat = flatten(state)
    for name, pat in stored.items():
        p = flatten(pat)
        if np.array_equal(flat, p):
            return name, "pattern"
    if include_negative:
        for name, pat in stored.items():
            p = flatten(pat)
            if np.array_equal(flat, -p):
                return name, "negative"
    return None, "spurious"


def closest_stored(
    state: np.ndarray, stored: dict[str, np.ndarray]
) -> tuple[str, int]:
    """Closest stored letter by Hamming distance, ignoring negatives."""
    flat = flatten(state)
    best_name = None
    best_d = int(flat.size + 1)
    for name, pat in stored.items():
        d = hamming_distance(flat, flatten(pat))
        if d < best_d:
            best_d = d
            best_name = name
    assert best_name is not None
    return best_name, best_d


def pattern_to_ascii(pattern: np.ndarray, on: str = "*", off: str = " ") -> str:
    """Render a 2D ±1 pattern as ASCII art (for logs)."""
    arr = np.asarray(pattern)
    return "\n".join("".join(on if v > 0 else off for v in row) for row in arr)


def pad_pattern(
    pattern: np.ndarray, target_shape: tuple[int, int], fill: int = -1
) -> np.ndarray:
    """Centrar un patrón en una grilla más grande rellenando con ``fill``."""
    arr = np.asarray(pattern, dtype=np.int8)
    h, w = arr.shape
    th, tw = target_shape
    if (h, w) == (th, tw):
        return arr.copy()
    if h > th or w > tw:
        raise ValueError(f"cannot pad {arr.shape} into {target_shape}")
    out = np.full(target_shape, fill, dtype=np.int8)
    y0 = (th - h) // 2
    x0 = (tw - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = arr
    return out


def resize_alphabet(
    letters: dict[str, np.ndarray], target_shape: tuple[int, int]
) -> dict[str, np.ndarray]:
    """Reescalar todo el alfabeto a ``target_shape`` centrando cada letra."""
    return {name: pad_pattern(pat, target_shape) for name, pat in letters.items()}


def resolve_letters_for_shape(
    base_path: str | Path,
    target_shape: tuple[int, int],
    *,
    letters_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Cargar letras para una dimensión: archivo dedicado o padding desde base."""
    if letters_path is not None:
        letters, shape = load_letters(letters_path, expected_shape=target_shape)
        return letters
    letters, shape = load_letters(base_path)
    if shape == target_shape:
        return letters
    return resize_alphabet(letters, target_shape)
