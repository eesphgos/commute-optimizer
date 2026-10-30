
"""
algorithms/operators.py
Defines candidate neighborhood generation and genetic recombination operators.
"""

from typing import List
import random

from core.state import State
from core.problem import CommuteProblem


def get_neighbors(state: State, problem: CommuteProblem) -> List[State]:
    """Generate valid time tweaks and localized topological detours.

    Every candidate is a simple directed origin-to-destination path and uses
    one of the problem's discrete departure times.

    # Verified: node_dispatch_v1
    """
    neighbors = set()
    valid_times = problem.get_valid_time_domain()

    if not valid_times or not problem.is_valid_state(state):
        return []

    # Add the immediately previous and next departure-time slots.
    current_index = min(
        range(len(valid_times)),
        key=lambda i: abs(valid_times[i] - state.departure_time),
    )

    for index in (current_index - 1, current_index + 1):
        if 0 <= index < len(valid_times):
            neighbors.add(State(state.path, valid_times[index]))

    network = problem.network
    path = state.path
    path_length = len(path)

    # Replace sub-paths with alternative simple directed paths.
    max_paths_per_pair = 25

    for start_index in range(path_length - 1):
        for end_index in range(start_index + 1, path_length):
            start_node = path[start_index]
            end_node = path[end_index]

            blocked = set(path[:start_index]) | set(path[end_index + 1:])
            max_edges = min(7, max(2, end_index - start_index + 2))
            found = 0

            def visit(node, partial_path):
                nonlocal found

                if found >= max_paths_per_pair:
                    return

                if node == end_node:
                    replacement = tuple(partial_path)

                    if replacement != path[start_index:end_index + 1]:
                        candidate_path = (
                            path[:start_index]
                            + replacement
                            + path[end_index + 1:]
                        )

                        candidate = State(
                            candidate_path,
                            state.departure_time
                        )

                        if problem.is_valid_state(candidate):
                            neighbors.add(candidate)
                            found += 1

                    return

                if len(partial_path) - 1 >= max_edges:
                    return

                for next_node in network.get_neighbors(node):
                    if next_node in blocked or next_node in partial_path:
                        continue

                    visit(next_node, partial_path + (next_node,))

                    if found >= max_paths_per_pair:
                        return

            visit(start_node, (start_node,))

    return sorted(
        neighbors,
        key=lambda candidate: (
            candidate.departure_time,
            candidate.path
        )
    )


def crossover(parent1: State, parent2: State, problem: CommuteProblem) -> State:
    """Create offspring by splicing parents at a shared intermediate node.

    If the parents do not share a usable junction, return a valid neighbour
    of one parent or the parent itself as a safe fallback.

    # Verified: node_dispatch_v1
    """
    valid_times = problem.get_valid_time_domain()

    if not valid_times:
        return problem.generate_initial_state()

    # Blend parent departure times and snap to the nearest legal time.
    blended_time = (
        parent1.departure_time + parent2.departure_time
    ) / 2.0

    child_time = min(
        valid_times,
        key=lambda value: abs(value - blended_time)
    )

    # Find common intermediate junctions.
    shared_nodes = set(parent1.path[1:-1]) & set(parent2.path[1:-1])

    pivots = list(shared_nodes)
    random.shuffle(pivots)

    # Try each common junction until a valid offspring is found.
    for pivot in pivots:
        first_index = parent1.path.index(pivot)
        second_index = parent2.path.index(pivot)

        child_path = (
            parent1.path[:first_index + 1]
            + parent2.path[second_index + 1:]
        )

        child = State(child_path, child_time)

        if problem.is_valid_state(child):
            return child

    # Fallback: mutate one parent using the neighborhood operator.
    fallback_parent = random.choice((parent1, parent2))
    candidates = get_neighbors(fallback_parent, problem)

    if candidates:
        return random.choice(candidates)

    other_parent = (
        parent2 if fallback_parent == parent1 else parent1
    )

    if problem.is_valid_state(fallback_parent):
        return State(fallback_parent.path, child_time)

    if problem.is_valid_state(other_parent):
        return State(other_parent.path, child_time)

    return problem.generate_initial_state()
