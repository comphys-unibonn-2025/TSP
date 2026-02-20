import numpy as np
from scipy.spatial.distance import pdist, squareform
from matplotlib.path import Path
import matplotlib.pyplot as plt

def polar_to_cartesian(r, phi):
    """
    converts polar coordinates (r, phi) to cartesian coordinates (x, y)
    """
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

def cartesian_to_polar(x, y):
    """
    converts cartesian coordinates (x, y) to polar coordinates (r, phi)
    """
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi

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
    :type traffic_factor: float, between 0 and 1 (1 means no traffic, 0 means full traffic)

    :param traffic_start_time: time at which traffic starts (in the same time units as the cost of traveling between nodes)
    :type traffic_start_time: float > 0
    """

    cost = 0
    current_traffic_factor = 1
    for i in range(len(route) - 1):

        if cost >= traffic_start_time:
            current_traffic_factor = traffic_factor

        cost += distance_matrix[route[i], route[i+1]] * (current_traffic_factor if (i in nodes_in_traffic and (i+1) in nodes_in_traffic) else 1)
    
    cost += distance_matrix[route[-1], route[0]] * (current_traffic_factor if (route[-1] in nodes_in_traffic and route[0] in nodes_in_traffic) else 1) # return to starting point

    return cost


def plot_tsp_route(nodes : np.array, route : np.array, traffic_polygon : np.array = None, output_path : str = None):
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

    plt.figure(figsize=(8, 8))

    # scatter nodes
    plt.scatter(nodes[:, 0], nodes[:, 1], color='black', label='Nodes', s=50)

    # starting point
    plt.plot(coords[0, 0], coords[0, 1], 'ro', label='Starting Point')

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
        width=0.006,
        color='blue'
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
        width=0.006,
        color='blue',
        label='TSP Route'
    )

    if traffic_polygon is not None:
        traffic_patch = plt.Polygon(traffic_polygon, color='red', alpha=0.3, label='Traffic Area')
        plt.gca().add_patch(traffic_patch)

    plt.gca().set_aspect('equal')
    plt.title('TSP Route')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()

    if output_path is not None:
        plt.savefig(output_path)

    plt.show()

def simmulated_annealing_tsp():
    pass

def nearest_neighbor_tsp(nodes, start_point_index = None):
    pass