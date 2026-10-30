
"""
algorithms/hill_climbing.py
Hill Climbing search with sideways move limits and random restarts.
"""

import random
from typing import Tuple

from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class HillClimbingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.max_sideways = max(0, int(self.params.get("max_sideways", 100)))
        self.max_restarts = max(0, int(self.params.get("max_restarts", 5)))
        self.max_iterations = max(0, int(self.params.get("max_iterations", 1000)))

    def _random_restart_state(self, start: State) -> State:
        """Create a different valid starting state by taking a random walk."""
        current = start
        visited = {current}

        walk_length = random.randint(1, 5)

        for _ in range(walk_length):
            candidates = [
                neighbor
                for neighbor in get_neighbors(current, self.problem)
                if neighbor not in visited
            ]

            if not candidates:
                break

            current = random.choice(candidates)
            visited.add(current)

        return current

    def solve(self) -> Tuple[State, float]:
        """
        Run steepest-descent hill climbing with sideways moves and restarts.

        The best global state is preserved across all restart attempts.
        """
        self.history = []

        initial_state = self.problem.generate_initial_state()
        best_state = initial_state
        best_cost = self.problem.evaluate(initial_state)
        self.history.append(best_cost)

        total_iterations = 0

        # First run starts from the baseline state.
        # Later runs start from randomized walks around that state.
        for restart in range(self.max_restarts + 1):
            if restart == 0:
                current = initial_state
            else:
                current = self._random_restart_state(initial_state)

            current_cost = self.problem.evaluate(current)
            visited = {current}
            sideways_used = 0

            if current_cost < best_cost:
                best_state = current
                best_cost = current_cost

            self.history.append(best_cost)

            while total_iterations < self.max_iterations:
                neighbors = [
                    neighbor
                    for neighbor in get_neighbors(current, self.problem)
                    if neighbor not in visited
                ]

                if not neighbors:
                    break

                scored_neighbors = [
                    (self.problem.evaluate(neighbor), neighbor)
                    for neighbor in neighbors
                ]

                # Prefer the best strictly improving neighbor.
                improving = [
                    item for item in scored_neighbors
                    if item[0] < current_cost - 1e-9
                ]

                if improving:
                    next_cost, next_state = min(
                        improving,
                        key=lambda item: item[0]
                    )
                    sideways_used = 0

                else:
                    # If no improvement exists, allow a limited
                    # number of cost-neutral sideways moves.
                    sideways = [
                        item for item in scored_neighbors
                        if abs(item[0] - current_cost) <= 1e-9
                    ]

                    if not sideways or sideways_used >= self.max_sideways:
                        break

                    next_cost, next_state = random.choice(sideways)
                    sideways_used += 1

                current = next_state
                current_cost = next_cost
                visited.add(current)
                total_iterations += 1

                if current_cost < best_cost:
                    best_state = current
                    best_cost = current_cost

                self.history.append(best_cost)

            if total_iterations >= self.max_iterations:
                break

        return best_state, best_cost
