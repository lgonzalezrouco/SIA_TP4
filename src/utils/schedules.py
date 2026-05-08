"""Learning-rate and neighborhood radius schedules for the SOM."""

from __future__ import annotations

from typing import Callable, Union

Schedule = Callable[[int, int], float]
ScheduleSpec = Union[int, float, dict]


def constant(value: float) -> Schedule:
    def _f(_step: int, _total: int) -> float:
        return float(value)

    return _f


def linear(initial: float, final: float) -> Schedule:
    """Linear interpolation between ``initial`` (at step 0) and ``final``
    (at step ``total - 1``)."""

    def _f(step: int, total: int) -> float:
        if total <= 1:
            return float(final)
        progress = min(max(step / (total - 1), 0.0), 1.0)
        return float(initial + (final - initial) * progress)

    return _f


def exponential(initial: float, final: float) -> Schedule:
    """Exponential decay ``initial * (final/initial) ** (step/(total-1))``.

    Equivalent to ``initial * exp(-step / tau)`` with
    ``tau = (total - 1) / ln(initial / final)``.
    """
    if initial <= 0 or final <= 0:
        raise ValueError("exponential schedule requires positive initial/final")
    ratio = final / initial

    def _f(step: int, total: int) -> float:
        if total <= 1:
            return float(final)
        progress = min(max(step / (total - 1), 0.0), 1.0)
        return float(initial * (ratio**progress))

    return _f


def inverse(initial: float, final: float) -> Schedule:
    """Inverse-time decay ``initial / (1 + step/T)`` with ``T`` chosen so
    that the value at the last step equals ``final``."""
    if initial <= 0 or final <= 0:
        raise ValueError("inverse schedule requires positive initial/final")
    if final >= initial:
        raise ValueError("inverse schedule requires final < initial")

    def _f(step: int, total: int) -> float:
        if total <= 1:
            return float(final)
        # Solve final = initial / (1 + (total-1)/T)  =>  T = (total-1) / (initial/final - 1)
        T = (total - 1) / (initial / final - 1.0)
        return float(initial / (1.0 + step / T))

    return _f


_FACTORIES: dict[str, Callable[..., Schedule]] = {
    "constant": lambda value, **_: constant(value),
    "linear": lambda initial, final, **_: linear(initial, final),
    "exponential": lambda initial, final, **_: exponential(initial, final),
    "inverse": lambda initial, final, **_: inverse(initial, final),
}


def parse(spec: ScheduleSpec) -> Schedule:
    """Build a Schedule from a number or a ``{"kind": ..., ...}`` dict."""
    if isinstance(spec, (int, float)):
        return constant(float(spec))
    if isinstance(spec, dict):
        kind = spec.get("kind", "constant")
        if kind not in _FACTORIES:
            raise ValueError(
                f"Unknown schedule kind '{kind}'. Available: {list(_FACTORIES)}"
            )
        params = {k: v for k, v in spec.items() if k != "kind"}
        return _FACTORIES[kind](**params)
    raise TypeError(f"Cannot parse schedule from {type(spec).__name__}")
