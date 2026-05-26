"""Discrete Hopfield network with Hebbian learning, sync/async recall and energy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RecallResult:
    """Output of a single recall run."""

    final_state: np.ndarray
    history: list[np.ndarray] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    converged: bool = False
    steps: int = 0
    cycle_detected: bool = False


class HopfieldNet:
    """Discrete Hopfield network with ±1 states.

    Weights are built with the standard Hebbian rule
    ``W = (1/N) * P @ P.T`` and ``diag(W) = 0``. Recall can be performed
    synchronously (all neurons updated at once) or asynchronously (one
    neuron at a time in random order). When ``h_i = 0`` the neuron keeps
    its previous state, as defined in the theory slides.
    """

    def __init__(self, n_neurons: int, seed: int | None = None):
        self.n_neurons = int(n_neurons)
        self.weights = np.zeros((self.n_neurons, self.n_neurons), dtype=float)
        self._rng = np.random.default_rng(seed)
        self.stored: np.ndarray | None = None

    # ------------------------------------------------------------- training
    def train(self, patterns: np.ndarray) -> None:
        """Store ``patterns`` (rows are flat ±1 vectors)."""
        P = np.asarray(patterns, dtype=float)
        if P.ndim != 2 or P.shape[1] != self.n_neurons:
            raise ValueError(
                f"patterns must have shape (p, {self.n_neurons}), got {P.shape}"
            )
        self.stored = P.copy()
        self.weights = (P.T @ P) / self.n_neurons
        np.fill_diagonal(self.weights, 0.0)

    # ------------------------------------------------------------- helpers
    def energy(self, state: np.ndarray) -> float:
        """Hopfield energy ``-0.5 * S^T W S`` (with ``w_ii = 0``)."""
        s = np.asarray(state, dtype=float)
        return float(-0.5 * s @ self.weights @ s)

    def _activate(self, h: np.ndarray, prev: np.ndarray) -> np.ndarray:
        """Sign with the convention ``h_i = 0 -> keep previous state``."""
        out = np.where(h > 0, 1.0, np.where(h < 0, -1.0, prev))
        return out

    # ------------------------------------------------------------- recall
    def recall(
        self,
        query: np.ndarray,
        max_steps: int = 50,
        async_update: bool = True,
        record_history: bool = True,
        rng: np.random.Generator | None = None,
    ) -> RecallResult:
        """Iterate until convergence (or ``max_steps``) and return the path."""
        rng = rng if rng is not None else self._rng
        s = np.asarray(query, dtype=float).copy()
        if s.shape != (self.n_neurons,):
            raise ValueError(
                f"query must have shape ({self.n_neurons},), got {s.shape}"
            )

        history: list[np.ndarray] = []
        energies: list[float] = []
        if record_history:
            history.append(s.copy())
            energies.append(self.energy(s))

        prev_states: list[np.ndarray] = [s.copy()]
        converged = False
        cycle = False
        steps_done = 0

        for step in range(1, max_steps + 1):
            if async_update:
                # one full sweep = one random permutation update
                new_s = s.copy()
                order = rng.permutation(self.n_neurons)
                for i in order:
                    h_i = self.weights[i] @ new_s
                    if h_i > 0:
                        new_s[i] = 1.0
                    elif h_i < 0:
                        new_s[i] = -1.0
                    # h_i == 0 -> keep
            else:
                h = self.weights @ s
                new_s = self._activate(h, s)

            steps_done = step
            if record_history:
                history.append(new_s.copy())
                energies.append(self.energy(new_s))

            if np.array_equal(new_s, s):
                converged = True
                s = new_s
                break

            # detect period-2 cycle (typical in synchronous mode)
            for past in prev_states[-3:-1]:
                if np.array_equal(new_s, past):
                    cycle = True
                    break

            s = new_s
            prev_states.append(s.copy())
            if cycle:
                break

        return RecallResult(
            final_state=s,
            history=history,
            energies=energies,
            converged=converged,
            steps=steps_done,
            cycle_detected=cycle,
        )


def run_hopfield(config):
    """Backward-compatible entry point (not used after migration)."""
    from .experiment import HopfieldExperiment

    HopfieldExperiment(config).run()
