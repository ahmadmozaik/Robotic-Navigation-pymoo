import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import glob

n_cities = 20
population_size = 60
n_epochs = 200
mutation_rate = 0.4

immigration_rate = 0.10
immigration_interval = 25
immigration_pool_multiplier = 5

alpha_weight = 0.7
beta_weight = 0.3

low_edge_penalty = 0
moderate_edge_penalty = 50
high_edge_penalty = 250

cities = None
distance_matrix = None
edge_penalty_matrix = None
_predefined_edge_penalty_matrix = None

frame_files_list = []
current_frame_idx = 0
img_display_ax = None
fig_viewer = None

def create_edge_penalty_matrix(num_cities, low_p, mod_p, high_p, distribution=(0.70, 0.20, 0.10)):
    """Generates a matrix of random penalties for all city-to-city edges.

        Creates a non-symmetric matrix where each directed edge (i, j) is
        randomly assigned a low, moderate, or high penalty value. The proportion
        of each penalty type across all edges is controlled by the `distribution`
        tuple.

        Args:
            num_cities (int): The number of cities in the graph (N).
            low_p (int or float): The penalty value for low-risk edges.
            mod_p (int or float): The penalty value for moderate-risk edges.
            high_p (int or float): The penalty value for high-risk edges.
            distribution (tuple, optional): A tuple of three floats (low, moderate,
                high) representing the proportion of each edge type.
                Defaults to (0.70, 0.20, 0.10).

        Returns:
            np.ndarray: An N x N matrix where matrix[i, j] is the penalty for
                        traveling from city `i` to city `j`.
    """
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
    num_cities = len(cities_array)
    matrix = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i == j:
                matrix[i, j] = 0
            else:
                matrix[i, j] = np.linalg.norm(cities_array[i] - cities_array[j])
    return matrix

def calculate_distance(path):
    dist = 0
    for i in range(len(path)):
        from_city = path[i]
        to_city = path[(i + 1) % len(path)]
        dist += distance_matrix[from_city, to_city]
    return dist

def calculate_path_penalty(path):
    penalty = 0
    for i in range(len(path)):
        from_city = path[i]
        to_city = path[(i + 1) % len(path)]
        penalty += edge_penalty_matrix[from_city, to_city]
    return penalty

def calculate_combined_fitness(distance, penalty, alpha, beta):
    """Calculates the weighted-sum fitness of a solution.

        This function combines the total distance and total penalty of a route into a
        single fitness value to be minimized. The formula used is:
        Fitness = (alpha * distance) + (beta * penalty).

        Args:
            distance (float): The total Euclidean distance of the path.
            penalty (float): The total accumulated penalty of the path.
            alpha (float): The weight assigned to the distance objective.
            beta (float): The weight assigned to the penalty objective.

        Returns:
            float: The final combined fitness score.
    """
    return (alpha * distance) + (beta * penalty)

def create_random_path_and_fitness():
    path_without_start = list(np.random.permutation(range(1, n_cities)))
    path = np.array([0] + path_without_start)
    dist = calculate_distance(path)
    pen = calculate_path_penalty(path)
    fit = calculate_combined_fitness(dist, pen, alpha_weight, beta_weight)
    return path, fit, dist, pen

def mutate(path):
    """Applies swap mutation to a given tour path.

        With a probability defined by the global `mutation_rate`, this function
        selects two random cities (excluding the starting city at index 0) in
        the path and swaps their positions.

        Args:
            path (np.ndarray): The tour path to mutate.

        Returns:
            np.ndarray: The potentially mutated path. If no mutation occurs,
                        it returns a copy of the original path.
    """
    new_path = path.copy()
    if np.random.rand() < mutation_rate:
        if len(path) > 2:
            idx_to_swap = range(1, len(path))
            if len(idx_to_swap) >= 2:
                i, j = np.random.choice(idx_to_swap, 2, replace=False)
                new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

def crossover(parent1_path, parent2_path):
    """Performs Ordered Crossover (OX) on two parent paths.

        A segment of the first parent is copied to the child, and the rest of
        the cities are filled in from the second parent in the order they appear.

        Args:
            parent1_path (np.ndarray): The first parent's tour path.
            parent2_path (np.ndarray): The second parent's tour path.

        Returns:
            np.ndarray: The resulting offspring's tour path.
    """
    size = len(parent1_path)
    child = [-1] * size
    start, end = sorted(np.random.choice(range(size), 2, replace=False))
    child[start:end] = parent1_path[start:end]
    fill_candidates = [city for city in parent2_path if city not in child[start:end]]
    ptr = 0
    for i in range(size):
        if child[i] == -1:
            if ptr < len(fill_candidates):
                child[i] = fill_candidates[ptr]
                ptr += 1
    if 0 not in child:
        if 0 in child:
            current_zero_idx = list(child).index(0)
            child[0], child[current_zero_idx] = child[current_zero_idx], child[0]
    elif child[0] != 0:
        try:
            zero_index = list(child).index(0)
            child[zero_index], child[0] = child[0], child[zero_index]
        except ValueError:
            pass
    return np.array(child)

def plot_route(path, epoch, current_combined_fitness, current_distance, current_penalty):
    """Generates and saves a detailed visual plot of a given TSP route.

        This function creates a visualization using Matplotlib that shows all city
        locations, the tour path, and highlights edges with moderate (orange) or
        high (red) penalties. The plot includes a title with current metrics and
        a text box with the full path sequence. The resulting plot is saved as a
        PDF file in the `frames/` directory.

        Args:
            path (np.ndarray): The tour path to visualize.
            epoch (int): The current epoch number, used in the plot title and filename.
            current_combined_fitness (float): The fitness score of the path.
            current_distance (float): The distance of the path.
            current_penalty (float): The penalty of the path.
    """
    plt.figure(figsize=(12, 10))
    plt.scatter(cities[:, 0], cities[:, 1], s=300, c='dodgerblue', zorder=2)

    for i in range(len(path)):
        from_city_idx = path[i]
        to_city_idx = path[(i + 1) % len(path)]
        from_city_coords = cities[from_city_idx]
        to_city_coords = cities[to_city_idx]
        edge_pen = edge_penalty_matrix[from_city_idx, to_city_idx]

        line_color = '#aaaaaa'
        line_width = 1.5
        show_penalty_text = False

        if edge_pen == high_edge_penalty:
            line_color = 'r-'
            line_width = 2.5
            show_penalty_text = True
        elif edge_pen == moderate_edge_penalty:
            line_color = 'orange'
            line_width = 2
            show_penalty_text = True

        plt.plot([from_city_coords[0], to_city_coords[0]],
                 [from_city_coords[1], to_city_coords[1]],
                 line_color, linewidth=line_width, zorder=1)

        if show_penalty_text:
            mid_x = (from_city_coords[0] + to_city_coords[0]) / 2
            mid_y = (from_city_coords[1] + to_city_coords[1]) / 2
            penalty_text = str(int(edge_pen))
            text_color = 'black'
            if edge_pen == high_edge_penalty:
                text_color = 'darkred'
            elif edge_pen == moderate_edge_penalty:
                text_color = 'darkorange'

            plt.text(mid_x, mid_y, penalty_text, color=text_color, fontsize=9,
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.8, pad=0.2, boxstyle='round,pad=0.3'),
                     zorder=4)

    for i_city, (x, y) in enumerate(cities):
        plt.text(x, y, str(i_city), fontsize=12, ha='center', va='center', color='white', weight='bold', zorder=3,
                 bbox=dict(facecolor='black', alpha=0.7, boxstyle='circle,pad=0.3'))

    title_text = (f"Epoch {epoch} | Fitness: {current_combined_fitness:.2f} "
                  f"(Dist: {current_distance:.2f}, Pen: {current_penalty:.2f})")
    plt.title(title_text, fontsize=14)
    plt.grid(True)
    plt.axis("equal")

    path_sequence_str = " -> ".join(map(str, path)) + " -> " + str(path[0])
    plt.figtext(0.5, 0.015, f"Path: {path_sequence_str}", wrap=True, horizontalalignment='center',
                fontsize=10,
                bbox=dict(facecolor='whitesmoke', alpha=0.7, pad=0.4, boxstyle='round,pad=0.3'))

    plt.subplots_adjust(bottom=0.08)
    os.makedirs("frames", exist_ok=True)
    plt.savefig(f"frames/epoch_{epoch:03d}.pdf")
    plt.close()

def update_frame_display(new_idx_offset=0):
    global current_frame_idx, img_display_ax, frame_files_list, fig_viewer

    if not frame_files_list:
        return

    current_frame_idx = (current_frame_idx + new_idx_offset) % len(frame_files_list)
    if current_frame_idx < 0:
        current_frame_idx += len(frame_files_list)

    img_path = frame_files_list[current_frame_idx]
    try:
        img = mpimg.imread(img_path)
        if img_display_ax is None:
            print("Error: Image display axes not initialized.")
            return

        img_display_ax.clear()
        img_display_ax.imshow(img)
        img_display_ax.axis('off')

        fname = os.path.basename(img_path)
        title = f"Frame Viewer: {fname} ({current_frame_idx + 1}/{len(frame_files_list)})"
        fig_viewer.suptitle(title, fontsize=12)
        fig_viewer.canvas.draw_idle()
    except FileNotFoundError:
        print(f"Error: Frame image not found at {img_path}")
    except Exception as e:
        print(f"Error loading/displaying frame {img_path}: {e}")

def on_key_press_viewer(event):
    global frame_files_list
    if not frame_files_list:
        return

    if event.key == 'right':
        update_frame_display(new_idx_offset=1)
    elif event.key == 'left':
        update_frame_display(new_idx_offset=-1)

def launch_interactive_viewer():
    global frame_files_list, current_frame_idx, img_display_ax, fig_viewer

    frame_files_list = sorted(glob.glob("frames/epoch_*.pdf"))

    if not frame_files_list:
        print("No frames found in 'frames' directory to display.")
        return

    current_frame_idx = 0
    fig_viewer, img_display_ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95)

    fig_viewer.canvas.mpl_connect('key_press_event', on_key_press_viewer)

    print("\nLaunching Interactive Frame Viewer...")
    print("Use LEFT and RIGHT arrow keys to navigate frames.")
    print(f"Found {len(frame_files_list)} frames.")

    update_frame_display()
    plt.show()

if __name__ == "__main__":
    use_fixed_scenario_input = input("Use fixed scenario (cities and penalties)? (y/n): ").strip().lower()
    use_fixed_scenario = use_fixed_scenario_input == 'y'

    if use_fixed_scenario:
        cities = np.array([
            [12, 68], [43, 87], [9, 38], [65, 24], [83, 73],
            [19, 12], [34, 92], [76, 28], [58, 49], [90, 11],
            [39, 66], [14, 81], [67, 36], [27, 59], [80, 53],
            [10, 45], [61, 69], [48, 15], [29, 77], [96, 34]
        ])
        print("\nFixed City Coordinates Used.")
    else:
        try:
            n_cities_input = int(input(f"Enter number of cities (default {n_cities}): ").strip())
            if n_cities_input > 1: n_cities = n_cities_input
        except ValueError:
            print(f"Invalid input, using default {n_cities} cities.")
        cities = np.random.randint(0, 100, size=(n_cities, 2))
        print(f"\nRandomly Generated {n_cities} City Coordinates.")

    n_cities = len(cities)
    distance_matrix = create_distance_matrix(cities)

    if use_fixed_scenario:
        if _predefined_edge_penalty_matrix is None or len(_predefined_edge_penalty_matrix) != n_cities:
            random.seed(42)
            _predefined_edge_penalty_matrix = create_edge_penalty_matrix(n_cities, low_edge_penalty,
                                                                         moderate_edge_penalty, high_edge_penalty)
            random.seed()
        edge_penalty_matrix = _predefined_edge_penalty_matrix
        print("Fixed Edge Penalties Used.")
    else:
        edge_penalty_matrix = create_edge_penalty_matrix(n_cities, low_edge_penalty, moderate_edge_penalty,
                                                         high_edge_penalty)
        print("Random Edge Penalties Generated for this run.")

    print_limit = min(5, n_cities)
    print(
        f"Edge Penalties Sample (first {print_limit}x{print_limit}):\n{edge_penalty_matrix[:print_limit, :print_limit]}")

    population = [create_random_path_and_fitness() for _ in range(population_size)]
    if not population:
        print("\nError: Initial population is empty.")
        exit()
    print("\nInitial randomized population (first path shown):", population[0][0])

    frame_interval_input = input("Generate frames every how many epochs? (e.g., 10; 0 to disable): ").strip()
    try:
        frame_interval = int(frame_interval_input)
        if frame_interval < 0: frame_interval = 0
    except ValueError:
        print("Invalid frame interval, disabling frame generation.")
        frame_interval = 0

    history_combined_fitness, history_distance, history_penalty = [], [], []
    print("\nStarting Evolution...")

    for epoch in range(n_epochs):
        population.sort(key=lambda x: x[1])
        best_individual = population[0]
        best_path, best_combined_fitness, best_distance, best_penalty = best_individual

        history_combined_fitness.append(best_combined_fitness)
        history_distance.append(best_distance)
        history_penalty.append(best_penalty)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(
                f"Epoch {epoch:03d}: Best Fitness = {best_combined_fitness:.2f} (Dist: {best_distance:.2f}, Pen: {best_penalty:.2f})")

        if frame_interval > 0 and epoch % frame_interval == 0:
            plot_route(best_path, epoch, best_combined_fitness, best_distance, best_penalty)

        if immigration_interval > 0 and epoch > 0 and epoch % immigration_interval == 0 and epoch != n_epochs - 1:
            num_immigrants = int(population_size * immigration_rate)
            if num_immigrants > 0:
                pool_size = num_immigrants * immigration_pool_multiplier
                immigrant_pool = sorted([create_random_path_and_fitness() for _ in range(pool_size)],
                                        key=lambda x: x[1])
                selected_immigrants = immigrant_pool[:num_immigrants]
                population[-num_immigrants:] = selected_immigrants
                print(f"Epoch {epoch:03d}: {len(selected_immigrants)} immigrants introduced.")
                population.sort(key=lambda x: x[1])

        next_gen = [best_individual]
        # Select parents from the top 20% of the population for breeding
        parent_pool_size = max(2, int(population_size * 0.2))
        parents_pool = population[:parent_pool_size] if len(population) >= parent_pool_size else population[:]

        while len(next_gen) < population_size:
            if len(parents_pool) >= 2:
                p1, p2 = random.sample(parents_pool, 2)
                child_path = mutate(crossover(p1[0], p2[0]))
                child_dist = calculate_distance(child_path)
                child_pen = calculate_path_penalty(child_path)
                child_fit = calculate_combined_fitness(child_dist, child_pen, alpha_weight, beta_weight)
                next_gen.append((child_path, child_fit, child_dist, child_pen))
            else:
                next_gen.append(create_random_path_and_fitness())
        population = next_gen

    population.sort(key=lambda x: x[1])
    final_path, final_combined_fitness, final_distance, final_penalty = population[0]

    if frame_interval > 0:
        plot_route(final_path, n_epochs, final_combined_fitness, final_distance, final_penalty)

    print("\n✅ Evolution complete.")
    print(f"Final best combined fitness: {final_combined_fitness:.2f}")
    print(f"Final best route distance: {final_distance:.2f}")
    print(f"Final best route penalty: {final_penalty:.2f}")
    print(f"Final best route: {final_path}")
    plt.figure(figsize=(12, 6))
    plt.plot(history_combined_fitness, color='purple', linewidth=2, label='Combined Fitness')
    plt.plot(history_distance, color='blue', linestyle='--', label='Distance')
    plt.plot(history_penalty, color='red', linestyle=':', label='Penalty')
    plt.title("Fitness, Distance, and Penalty Over Generations")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("multi_objective_convergence_plot.pdf")
    print("\nConvergence plot saved as 'multi_objective_convergence_plot.pdf'")
    plt.show(block=False)

    if frame_interval > 0:
        launch_interactive_viewer()
    else:
        print("\nFrame generation was disabled, skipping interactive viewer.")
        if not plt.get_fignums():
            pass
        else:
            print("Convergence plot is open. Close it to exit script if it's the last window.")


    print("\nScript finished.")
