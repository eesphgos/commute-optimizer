
"""
algorithms/genetic_algorithm.py
Population-based optimization using selection, recombination, and mutation.
"""

import math
import random
from typing import Tuple

from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, crossover


class GeneticAlgorithmSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.population_size: int = self.params.get("population_size", 30)
        self.generations: int = self.params.get("generations", 100)
        self.crossover_rate: float = self.params.get("crossover_rate", 0.8)
        self.mutation_rate: float = self.params.get("mutation_rate", 0.2)
        self.elitism: int = self.params.get("elitism", 2)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Genetic Algorithm optimization.

        Uses tournament selection, crossover, stochastic mutation,
        and elitism to minimize commute travel time.
        """
        self.history = []
        self.iterations = 0

        population_size = max(2, int(self.population_size))
        generations = max(0, int(self.generations))

        crossover_rate = min(1.0, max(0.0, float(self.crossover_rate)))
        mutation_rate = min(1.0, max(0.0, float(self.mutation_rate)))

        elitism = min(
            population_size,
            max(0, int(self.elitism))
        )

        def fitness(cost: float) -> float:
            if not math.isfinite(cost):
                return 0.0
            return 1.0 / (1.0 + max(0.0, cost))

        initial_state = self.problem.generate_initial_state()

        if not self.problem.is_valid_state(initial_state):
            raise ValueError("The initial state is invalid.")

        population = [initial_state]
        seen = {initial_state}

        attempts = 0
        max_attempts = population_size * 15

        while len(population) < population_size and attempts < max_attempts:
            attempts += 1

            parent = random.choice(population)
            candidates = [
                neighbor
                for neighbor in get_neighbors(parent, self.problem)
                if (
                    self.problem.is_valid_state(neighbor)
                    and neighbor not in seen
                )
            ]

            if not candidates:
                continue

            child = random.choice(candidates)
            population.append(child)
            seen.add(child)

        while len(population) < population_size:
            population.append(random.choice(population))

        costs = [
            self.problem.evaluate(state)
            for state in population
        ]

        best_index = min(
            range(len(population)),
            key=lambda index: costs[index]
        )

        best_state = population[best_index]
        best_cost = costs[best_index]

        self.history.append(best_cost)

        def select_parent() -> State:
            tournament_size = min(3, len(population))

            indices = random.sample(
                range(len(population)),
                tournament_size
            )

            winner = max(
                indices,
                key=lambda index: fitness(costs[index])
            )

            return population[winner]

        for _ in range(generations):

            ranked_indices = sorted(
                range(len(population)),
                key=lambda index: costs[index]
            )

            next_population = [
                population[index]
                for index in ranked_indices[:elitism]
            ]

            while len(next_population) < population_size:

                parent1 = select_parent()
                parent2 = select_parent()

                if random.random() < crossover_rate:
                    child = crossover(
                        parent1,
                        parent2,
                        self.problem
                    )
                else:
                    child = parent1

                if (
                    not isinstance(child, State)
                    or not self.problem.is_valid_state(child)
                ):
                    child = parent1

                if random.random() < mutation_rate:
                    mutation_candidates = [
                        neighbor
                        for neighbor in get_neighbors(
                            child,
                            self.problem
                        )
                        if self.problem.is_valid_state(neighbor)
                    ]

                    if mutation_candidates:
                        child = random.choice(mutation_candidates)

                if not self.problem.is_valid_state(child):
                    child = parent1

                next_population.append(child)

            population = next_population

            costs = [
                self.problem.evaluate(state)
                for state in population
            ]

            generation_best_index = min(
                range(len(population)),
                key=lambda index: costs[index]
            )

            generation_best_state = population[generation_best_index]
            generation_best_cost = costs[generation_best_index]

            if generation_best_cost < best_cost:
                best_state = generation_best_state
                best_cost = generation_best_cost

            self.history.append(best_cost)

            self.iterations += 1

        return best_state, best_cost
