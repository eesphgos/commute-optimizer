
"""
algorithms/beam_search.py
Local Beam Search maintaining K parallel candidate states.
"""

from typing import Tuple

from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class LocalBeamSearchSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.beam_width: int = self.params.get("beam_width", 5)
        self.max_iterations: int = self.params.get("max_iterations", 150)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Local Beam Search.

        Maintains multiple candidate states, pools their successors,
        ranks them collectively, and retains the best beam_width states.
        """
        self.history = []
        self.iterations = 0

        initial_state = self.problem.generate_initial_state()
        initial_cost = self.problem.evaluate(initial_state)

        beam = [initial_state]
        visited = {initial_state}

        best_state = initial_state
        best_cost = initial_cost

        self.history.append(best_cost)

        beam_width = max(1, int(self.beam_width))

        for _ in range(max(0, int(self.max_iterations))):

            pooled = set()

            for state in beam:
                for neighbor in get_neighbors(state, self.problem):
                    if (
                        neighbor not in visited
                        and self.problem.is_valid_state(neighbor)
                    ):
                        pooled.add(neighbor)

            if not pooled:
                break

            scored = [
                (self.problem.evaluate(candidate), candidate)
                for candidate in pooled
            ]

            scored.sort(
                key=lambda item: (
                    item[0],
                    len(item[1].path),
                    item[1].path,
                    item[1].departure_time,
                )
            )

            next_beam = []
            selected = set()
            selected_paths = set()

            for cost, candidate in scored:
                if candidate.path not in selected_paths:
                    next_beam.append(candidate)
                    selected.add(candidate)
                    selected_paths.add(candidate.path)

                    if len(next_beam) == beam_width:
                        break

            if len(next_beam) < beam_width:
                for cost, candidate in scored:
                    if candidate not in selected:
                        next_beam.append(candidate)
                        selected.add(candidate)

                        if len(next_beam) == beam_width:
                            break

            visited.update(pooled)

            beam = next_beam
            self.iterations += 1

            generation_best_cost, generation_best_state = scored[0]

            improved = generation_best_cost < best_cost - 1e-9

            if improved:
                best_state = generation_best_state
                best_cost = generation_best_cost

            self.history.append(best_cost)

            if not improved:
                break

        return best_state, best_cost
