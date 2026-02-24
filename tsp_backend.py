import numpy as np
from scipy.spatial.distance import pdist, squareform
from matplotlib.path import Path
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,          # Use LaTeX rendering (requires LaTeX installed)
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10              # Set all text to 10 pt
})

def polar_to_cartesian(r, phi):
    """
    converts polar coordinates (r, phi) to cartesian coordinates (x, y)
    """
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.column_stack([x, y])

def cartesian_to_polar(x, y):
    """
    converts cartesian coordinates (x, y) to polar coordinates (r, phi)
    """
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.column_stack([r, phi])

def metric(p1 : np.ndarray, p2 : np.ndarray) -> float:
    """
    calculates the eucledian distance between two points p1 and p2
    note: p1 and p2 need to be in cartesian coordinates
    """
    if p1.shape != p2.shape:
        raise Exception('The points p1 and p2 need to have the same shape')

    return np.sqrt(np.sum(np.square(p1 - p2)))

def make_distance_matrix(nodes : np.ndarray, metric_func = metric) -> np.ndarray:
    """
    calculates the distance matrix D_{i,j} of the nodes, 
    where D_{i,j} denotes the distance of node i to node j, using the given metric
    """
    D = squareform(pdist(nodes, metric=metric_func))
    return D

def check_if_in_polygon(point : np.ndarray, polygon : np.ndarray) -> bool:
    """
    checks if a point is inside a polygon defined by its vertices
    note: the point and the polygon vertices need to be in cartesian coordinates
    """
    if point.shape != polygon.shape[1:]:
        raise Exception('The point and the polygon vertices need to have the same shape')

    path = Path(polygon)
    return path.contains_point(point)

def calculate_route_cost(nodes : np.array, route : np.array, nodes_in_traffic : np.array, distance_matrix : np.array, traffic_factor : float = 1, traffic_start_time : float = 0) -> float:
    """
    calculates the cost of a given route, with a given distance/cost matrix and traffic factor (which can be time-dependent)
    
    :param nodes: array of coordinates of the N nodes
    :type nodes: np.ndarray (N, d), N: number of nodes, d: dimension
    
    :param route: array of node indices representing the order of visiting the nodes
    :type route: np.ndarray (N,), values in [0, N-1]

    :param nodes_in_traffic: array of node indices which are affected by traffic
    :type nodes_in_traffic: np.ndarray (M,), values in [0, N-1], M: number of nodes affected by traffic
    
    :param distance_matrix: matrix of distances/costs between the nodes
    :type distance_matrix: np.ndarray (N, N)

    :param traffic_factor: factor by which the cost of traveling between nodes is multiplied during traffic time
    :type traffic_factor: float >= 1 (1 means no traffic, >1 means traffic)

    :param traffic_start_time: time at which traffic starts (in the same time units as the cost of traveling between nodes)
    :type traffic_start_time: float >= 0
    """

    cost = 0
    current_traffic_factor = 1
    previous_stop_in_traffic = False
    for i in range(len(route) - 1):

        if cost >= traffic_start_time:
            current_traffic_factor = traffic_factor

            if not previous_stop_in_traffic and (route[i] in nodes_in_traffic or route[i+1] in nodes_in_traffic):
                current_traffic_factor = current_traffic_factor * (cost - traffic_start_time) / distance_matrix[route[i], route[i+1]]
            
            previous_stop_in_traffic = True


        cost += distance_matrix[route[i], route[i+1]] * (current_traffic_factor if (i in nodes_in_traffic and (i+1) in nodes_in_traffic) else 1)
    
    cost += distance_matrix[route[-1], route[0]] * (current_traffic_factor if (route[-1] in nodes_in_traffic and route[0] in nodes_in_traffic) else 1) # return to starting point

    return cost


def plot_tsp_route(nodes : np.array, 
                   route : np.array, 
                   traffic_polygon : np.array = None, 
                   output_path : str = None, 
                   cost : float = None,
                   arrow_width = 0.006,
                   arrow_color = 'lightblue',
                   node_dot_size = 50,
                   node_color='black',
                   want_axis=True):
    """plots the TSP route defined by the order of visiting the nodes
    
    :param nodes: array of coordinates of the N nodes
    :type nodes: np.ndarray (N, d), N: number of nodes, d: dimension
    
    :param route: array of node indices representing the order of visiting the nodes
    :type route: np.ndarray (N,), values in [0, N-1]

    :param traffic_polygon: array of coordinates of the vertices of the polygon representing the traffic area (optional)
    :type traffic_polygon: np.ndarray (M, d), M: number of vertices, d: dimension

    :param output_path: path to save the plot (optional)
    :type output_path: str
    """

    coords = nodes[route]

    plt.figure(figsize=(3.4, 4))

    # scatter nodes
    plt.scatter(nodes[:, 0], nodes[:, 1], color=node_color, label='Nodes',zorder = 1, s=node_dot_size)

    # starting point
    plt.scatter(coords[0, 0], coords[0, 1], color='red', label='Starting Point', zorder = 2, s=node_dot_size)

    # direction vectors
    dx = coords[1:, 0] - coords[:-1, 0]
    dy = coords[1:, 1] - coords[:-1, 1]

    plt.quiver(
        coords[:-1, 0],   # arrow start x
        coords[:-1, 1],   # arrow start y
        dx,               # x direction
        dy,               # y direction
        angles='xy',
        scale_units='xy',
        scale=1,
        width=arrow_width,
        color=arrow_color
    )

    # optional: close the tour
    dx_last = coords[0, 0] - coords[-1, 0]
    dy_last = coords[0, 1] - coords[-1, 1]

    plt.quiver(
        coords[-1, 0],
        coords[-1, 1],
        dx_last,
        dy_last,
        angles='xy',
        scale_units='xy',
        scale=1,
        width=arrow_width,
        color=arrow_color,
        label='TSP Route'
    )

    if traffic_polygon is not None:
        traffic_patch = plt.Polygon(traffic_polygon, color='red', alpha=0.3, label='Traffic Area')
        plt.gca().add_patch(traffic_patch)

    plt.gca().set_aspect('equal')

    plt.title('TSP Route')
    if not cost is None:
        plt.title(f"TSP Route, Cost: {cost:.2f}")

    if want_axis:
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
    else:
        plt.axis(False)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight')

    plt.show()

def simmulated_annealing_tsp(
        nodes : np.array, 
        traffic_polygon : np.array = None,
        traffic_factor : float = 1,
        traffic_start_time : float = 0,
        temperatures : list = [0.1, 0.05, 0.01, 0.001], 
        start_point_index = None, 
        rejection_threshold : list = [20, 200, 2000, 10000], 
        max_iter_per_temperature : int = 100000,
        plot_cost : bool = False,
        plot_cost_out_path : str = None) -> np.array:
    """
    simulated annealing algorithm for the time-dependent traveling salesman problem
    
    :param nodes: array of coordinates of the N nodes
    :type nodes: np.ndarray (N, d), N: number of nodes, d: dimension

    :param traffic_polygon: array of coordinates of the vertices of the polygon representing the traffic area (optional)
    :type traffic_polygon: np.ndarray (M, d), M: number of vertices, d: dimension

    :param traffic_factor: factor by which the cost of traveling between nodes is multiplied during traffic time
    :type traffic_factor: float >= 1 (1 means no traffic, >1 means traffic)

    :param traffic_start_time: time at which traffic starts (in the same time units as the cost of traveling between nodes)
    :type traffic_start_time: float >= 0

    :param temperatures: list of decreasing temperatures for the annealing
    :type temperatures: list of float

    :param start_point_index: index of the node to start the route from (optional)
    :type start_point_index: int in [0, N-1]

    :param rejection_threshold: list of maximum number of rejections per temperature (optional)
    :type rejection_threshold: list of int, same length as temperatures

    :param max_iter_per_temperature: maximum number of iterations per temperature (optional)
    :type max_iter_per_temperature: int

    :param plot_cost: whether to plot the cost evolution (optional)
    :type plot_cost: bool
    
    :param plot_cost_out_path: path to save the cost plot (optional)
    :type plot_cost_out_path: str

    :return: best route found
    :rtype: np.ndarray (N,), values in [0, N-1]
    """

    N = len(nodes)
    distances = make_distance_matrix(nodes)

    # determine which nodes are affected by traffic
    if traffic_polygon is not None:
        nodes_in_traffic = np.array([i for i in range(N) if check_if_in_polygon(nodes[i], traffic_polygon)])
    else:
        nodes_in_traffic = np.array([])

    # randomly generate starting permutation and place starting_point_index in the front (w/o duplicates)
    if start_point_index is None:
        starting_permutation = np.random.permutation(N)
    else:
        starting_permutation = np.concatenate((np.array([start_point_index]), np.random.permutation(np.arange(N)[np.arange(N) != start_point_index])))
        
    # step 1
    current_route = starting_permutation
    current_cost = calculate_route_cost(nodes, current_route, nodes_in_traffic, distances, traffic_factor, traffic_start_time)
    costs = [current_cost]

    trial_route = np.zeros((N, 2))
    trial_cost = 0.0

    num_consec_rejections = 0
    num_iter = 0
    temp_change_points = []
    # step 2
    for temp, rej_thresh in zip(temperatures, rejection_threshold):

        temp_change_points.append(num_iter)

        # print(f"Temperature: {temp}, Current Cost: {current_cost}")
        # print(f"Current route: {current_route}")
        # print(f"num consec rejections: {num_consec_rejections}, num iter: {num_iter}")

        num_consec_rejections = 0

        num_iter = 0
        while num_consec_rejections < rej_thresh and num_iter < max_iter_per_temperature:

            # step 3
            for i in range(0, N):

                j = np.random.choice(np.delete(np.arange(0, N), i))
                
                # step 4
                if j < i: # swap i and j if necessary
                    i, j = j, i
                
                trial_route = current_route.copy()
                trial_route[i:j+1] = trial_route[i:j+1][::-1] # invert order of ith to jth entry

                if not start_point_index is None and trial_route[0] != start_point_index: # ensure starting point is at the front
                    trial_route[trial_route == start_point_index] = trial_route[0]
                    trial_route[0] = start_point_index

                # step 5
                trial_cost = calculate_route_cost(nodes, trial_route, nodes_in_traffic, distances, traffic_factor, traffic_start_time)

                # step 6 & 7
                if trial_cost < current_cost:
                    current_route = trial_route.copy()
                    num_consec_rejections = 0
                    continue

                num_consec_rejections += 1

                x = np.random.rand()
                if x < np.exp((current_cost - trial_cost) / temp):
                    current_route = trial_route.copy()
                    num_consec_rejections = 0
                
                current_cost = calculate_route_cost(nodes, current_route, nodes_in_traffic, distances, traffic_factor, traffic_start_time)
                costs.append(current_cost)
                num_iter += 1
    
    if plot_cost:
        plt.figure(figsize=(3.4,5))
        plt.title(f"TSP cost evolution, \nfinal cost: {costs[-1]:.2f}")
        for (i, temp_change_points) in enumerate(temp_change_points):
            plt.plot([temp_change_points, temp_change_points], [min(costs) - 0.5, max(costs)], alpha = 0.4, color='grey') #, label=rf'{temperature[i]} $$\unit{{\kelvin}}$$')
        plt.plot(costs, color='black', label='TSP Length')
        # plt.xlim(0.99*len(length_arr), len(length_arr))
        # plt.ylim(6, 13)
        #plt.xscale("log")
        plt.xlabel('Iteration')
        plt.ylabel('TSP Length')

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        plt.tight_layout()

        if not plot_cost_out_path is None:
            plt.savefig(plot_cost_out_path)
        plt.show()

    return current_route, costs[-1]

def nearest_neighbor_tsp(nodes, start_point_index = None):
    pass