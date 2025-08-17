import numpy as np
import random
import matplotlib.pyplot as plt

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.ox import OrderedCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.mutation.inversion_mutation import InversionMutation


# --- 1. Problem Definition (Pymoo's approach) ---

class TSPWithPenaltiesProblem(Problem):
    """
    A custom problem class for the TSP with two objectives:
    - Total Distance
    - Total Penalty
    """

    def __init__(self, n_cities, distance_matrix, penalty_matrix):
        # The problem has n_cities variables (the permutation of cities),
        # 2 objectives, and no constraints.
        super().__init__(n_var=n_cities, n_obj=2, n_constr=0, type_var=np.int)
        self.distance_matrix = distance_matrix
        self.penalty_matrix = penalty_matrix

    def _evaluate(self, X, out, *args, **kwargs):
        # X is a NumPy array where each row represents a tour (individual)
        distances = np.zeros(X.shape[0])
        penalties = np.zeros(X.shape[0])

        for i, path in enumerate(X):
            distances[i] = self._calculate_distance(path)
            penalties[i] = self._calculate_path_penalty(path)

        # We assign the two objectives to the output dictionary 'out' under the key "F"
        out["F"] = np.column_stack([distances, penalties])

    def _calculate_distance(self, path):
        # This function is adapted from the original code
        dist = 0
        for i in range(len(path)):
            from_city = path[i]
            to_city = path[(i + 1) % len(path)]
            dist += self.distance_matrix[from_city, to_city]
        return dist

    def _calculate_path_penalty(self, path):
        # This function is adapted from the original code
        penalty = 0
        for i in range(len(path)):
            from_city = path[i]
            to_city = path[(i + 1) % len(path)]
            penalty += self.penalty_matrix[from_city, to_city]
        return penalty


# --- 2. Helper Functions (Adapted from initialCode.txt) ---

def create_edge_penalty_matrix(num_cities, low_p, mod_p, high_p, distribution=(0.70, 0.20, 0.10)):
    """Generates a matrix of random penalties for all city-to-city edges."""
    matrix = np.zeros((num_cities, num_cities))
    all_edges = []
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                all_edges.append((i, j))
    random.shuffle(all_edges)
    num_total_edges = len(all_edges)
    num_low = int(num_total_edges * distribution[0])
    num_mod = int(num_total_edges * distribution[1])

    for k in range(num_low):
        if all_edges:
            i, j = all_edges.pop()
            matrix[i, j] = low_p
    for k in range(num_mod):
        if all_edges:
            i, j = all_edges.pop()
            matrix[i, j] = mod_p
    while all_edges:
        i, j = all_edges.pop()
        matrix[i, j] = high_p
    return matrix


def create_distance_matrix(cities_array):
    """Calculates the Euclidean distance between all pairs of cities."""
    num_cities = len(cities_array)
    matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                matrix[i, j] = np.linalg.norm(cities_array[i] - cities_array[j])
    return matrix


# --- 3. Main Execution ---

if __name__ == "__main__":

    # Define problem parameters
    n_cities = 20
    low_edge_penalty = 0
    moderate_edge_penalty = 50
    high_edge_penalty = 250

    # Fixed city coordinates from the original code for a reproducible scenario
    cities = np.array([
        [12, 68], [43, 87], [9, 38], [65, 24], [83, 73],
        [19, 12], [34, 92], [76, 28], [58, 49], [90, 11],
        [39, 66], [14, 81], [67, 36], [27, 59], [80, 53],
        [10, 45], [61, 69], [48, 15], [29, 77], [96, 34]
    ])

    # Generate the distance and penalty matrices
    distance_matrix = create_distance_matrix(cities)
    edge_penalty_matrix = create_edge_penalty_matrix(n_cities, low_edge_penalty, moderate_edge_penalty,
                                                     high_edge_penalty)

    # Instantiate the problem class
    problem = TSPWithPenaltiesProblem(n_cities, distance_matrix, edge_penalty_matrix)

    # --- 4. Algorithm Setup (using Pymoo) ---

    # NSGA-II is a well-suited algorithm for multi-objective problems
    algorithm = NSGA2(
        pop_size=100,
        # Use a random permutation as the initial population
        sampling=PermutationRandomSampling(),
        # Use Ordered Crossover, suitable for permutations
        crossover=OrderedCrossover(),
        # Use Inversion Mutation, a common operator for permutation problems
        mutation=InversionMutation(),
        eliminate_duplicates=True
    )

    # --- 5. Run the Optimization ---

    # The minimize function replaces the original evolution loop
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 200),  # Run for 200 generations
        seed=1,
        verbose=True
    )

    # --- 6. Analysis and Visualization ---

    if res.F is not None:
        print("\n✅ Evolution complete. Final Pareto-optimal front found.")

        # The Pareto front contains the best trade-off solutions found
        # Each point represents a solution with its distance and penalty values
        Scatter(title="Pareto Front").add(res.F).show()

        # You can also access the best solutions found
        best_solutions = res.X
        best_objectives = res.F

        # Example: Print the first few solutions from the final population
        for i in range(min(5, len(best_solutions))):
            print(f"Solution {i + 1}:")
            print(f"  Path: {best_solutions[i]}")
            print(f"  Distance: {best_objectives[i][0]:.2f}")
            print(f"  Penalty: {best_objectives[i][1]:.2f}")
    else:
        print("\nEvolution finished, but no non-dominated solutions were found.")