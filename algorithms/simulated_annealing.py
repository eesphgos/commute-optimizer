
"""
algorithms/simulated_annealing.py
Stochastic optimization governed by a temperature schedule.
"""

import math
import random
from typing import Tuple

from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class SimulatedAnnealingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.initial_temp: float = self.params.get("initial_temp", 100.0)
        self.cooling_rate: float = self.params.get("cooling_rate", 0.995)
        self.min_temp: float = self.params.get("min_temp", 1e-3)
        self.max_iterations: int = self.params.get("max_iterations", 2000)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Simulated Annealing optimization.

        Improvements are always accepted. Worse moves are accepted
        probabilistically using the Metropolis criterion. Temperature
        decreases exponentially after each iteration.
        """
        self.history = []
        self.iterations = 0

        current_state = self.problem.generate_initial_state()
        current_cost = self.problem.evaluate(current_state)

        best_state = current_state
        best_cost = current_cost

        temperature = self.initial_temp

        self.history.append(best_cost)

        for _ in range(self.max_iterations):

            if temperature <= self.min_temp:
                break

            neighbors = get_neighbors(current_state, self.problem)

            if not neighbors:
                break

            candidate_state = random.choice(neighbors)
            candidate_cost = self.problem.evaluate(candidate_state)

            delta = candidate_cost - current_cost

            if delta <= 0:
                accept = True

            elif temperature > 0:
                probability = math.exp(-delta / temperature)
                accept = random.random() < probability

            else:
                accept = False

            if accept:
                current_state = candidate_state
                current_cost = candidate_cost

            if current_cost < best_cost:
                best_state = current_state
                best_cost = current_cost

            self.history.append(best_cost)

            self.iterations += 1

            temperature *= self.cooling_rate

        return best_state, best_cost
