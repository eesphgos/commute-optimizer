"""
algorithms/hill_climbing.py
Hill Climbing search with sideways move limits and random restarts.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class HillClimbingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.max_sideways: int = self.params.get("max_sideways", 100)
        self.max_restarts: int = self.params.get("max_restarts", 5)

    
    def solve(self) -> Tuple[State, float]:
        """Run basic steepest-descent hill climbing."""
        self.history = []

        current = self.problem.generate_initial_state()
        current_cost = self.problem.evaluate(current)

        best_state = current
        best_cost = current_cost
        self.history.append(best_cost)

        max_iterations = int(self.params.get("max_iterations", 1000))

        for _ in range(max_iterations):
            neighbors = get_neighbors(current, self.problem)

            if not neighbors:
                break

            next_state = min(
                neighbors,
                key=lambda state: self.problem.evaluate(state)
            )
            next_cost = self.problem.evaluate(next_state)

            if next_cost >= current_cost - 1e-9:
                break

            current = next_state
            current_cost = next_cost
            current = next_state

            if current_cost < best_cost:
                best_state = current
                best_cost = current_cost

            self.history.append(best_cost)

        return best_state, best_cost

        raise NotImplementedError("Implement solve() in algorithms/hill_climbing.py")
