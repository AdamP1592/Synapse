import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import Patch
from shapely.geometry import Point, Polygon
from shapely import centroid, area

# Global holders for generated neuron polygons (for plotting and overlap computation)
axon_polys = []
dendrite_polys = []

alpha = 5# scaling parameter for probabilistic connection


class connection:
    """
    Represents a synaptic connection between neurons defined by an overlapping polygon.
    
    Attributes:
        hosts (list): List of indices for pre-synaptic neurons.
        connections (list): List of indices for post-synaptic neurons.
        connection_poly (Polygon): A Shapely Polygon representing the overlap area.
    """
    def __init__(self):
        self.hosts = []
        self.connections = []
        self.connection_poly = None

    def get_center(self) -> list:
        """
        Returns the center coordinates of the connection polygon.
        
        :return: [x, y] coordinates of the centroid.
        """
        center_point = self.connection_poly.centroid
        return [center_point.x, center_point.y]

    def get_area(self) -> float:
        """
        Returns the area of the connection polygon.
        
        :return: Area as a float.
        """
        return area(self.connection_poly)

    def add_host(self, host):
        """
        Adds a pre-synaptic neuron index.
        
        :param host: Pre-synaptic neuron index.
        """
        self.hosts.append(host)

    def add_connection(self, connection):
        """
        Adds a post-synaptic neuron index.
        
        :param connection: Post-synaptic neuron index.
        """
        self.connections.append(connection)

    def copy(self):
        """
        Creates a copy of the connection object.
        
        :return: A new connection object with the same hosts, connections, and polygon.
        """
        new_con = connection()
        new_con.hosts = self.hosts.copy()
        new_con.connections = self.connections.copy()
        new_con.connection_poly = self.connection_poly
        return new_con

    def __str__(self) -> str:
        return "[pre_synaptic_neurons: {}, post_synaptic_neurons: {}, poly_area: {}]".format(
            str(self.hosts), str(self.connections), str(self.get_area())
        )


# ------------------- Synapse Generator Functions -------------------

def generate_semicircle_polygon(center: Point, radius: float, theta1: float, theta2: float, num_points: int = 100) -> Polygon:
    """
    Generates a semicircular polygon (as a Shapely Polygon) given a center, radius, and angle range.
    
    :param center: A Shapely Point representing the center.
    :param radius: Radius of the semicircle.
    :param theta1: Starting angle (in radians).
    :param theta2: Ending angle (in radians).
    :param num_points: Number of points along the arc.
    :return: A Shapely Polygon.
    """
    angles = np.linspace(theta1, theta2, num_points)
    points = [(center.x + radius * np.cos(angle), center.y + radius * np.sin(angle)) for angle in angles]
    points.append((center.x, center.y))  # Close the polygon by adding the center point.
    return Polygon(points)


def plot_point(ax, point: Point, color: str, alpha: float = 1, annotation: str = "", border_width: int = 0):
    """
    Plots a point on the given axes.
    
    :param ax: Matplotlib axes.
    :param point: A Shapely Point.
    :param color: Color code (e.g., "#ff0000").
    :param alpha: Transparency level.
    :param annotation: Text to annotate the point.
    """
    ax.plot(point.x, point.y, "o", color=color, alpha=alpha, markeredgecolor='black', markeredgewidth=border_width)
    ax.annotate(annotation, xy=(point.x, point.y))


def plot_filled_polygon(ax, polygon: Polygon, color: str, alpha: float = 0.5, linestyle: str = 'solid', annotation: str = "", annotation_color: str = "white"):
    """
    Plots a filled polygon on the given axes.
    
    :param ax: Matplotlib axes.
    :param polygon: A Shapely Polygon.
    :param color: Fill color.
    :param alpha: Transparency.
    :param linestyle: Border line style.
    :param annotation: Optional text annotation.
    :param annotation_color: Color for annotation text.
    """
    mpl_poly = MplPolygon(list(polygon.exterior.coords), closed=True, color=color, alpha=alpha, linestyle=linestyle)
    ax.add_patch(mpl_poly)
    center = centroid(polygon)
    ax.annotate(annotation, xy=(center.x, center.y), color=annotation_color)


def generate_neuron_polys(soma_points: list, r_dendrite: float, r_axon: float, dendrite_thetas: list, axon_thetas: list) -> list:
    """
    Generates axonal and dendritic semicircular polygons for a list of neuron soma points.
    
    :param soma_points: List of Shapely Points representing neuron positions.
    :param r_dendrite: Radius for dendritic polygons.
    :param r_axon: Radius for axonal polygons.
    :param dendrite_thetas: List of (theta1, theta2) tuples for dendritic arcs.
    :param axon_thetas: List of (theta1, theta2) tuples for axonal arcs.
    :return: List containing two lists: [axon_polys, dendrite_polys].
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    dendrite_polys = []
    axon_polys = []
    
    for i in range(len(soma_points)):
        color_index = i % len(colors)
        center = soma_points[i]
        theta1_dendrite, theta2_dendrite = dendrite_thetas[i]
        theta1_axon, theta2_axon = axon_thetas[i]
        
        dendrite_poly = generate_semicircle_polygon(center, r_dendrite, theta1_dendrite, theta2_dendrite)
        axon_poly = generate_semicircle_polygon(center, r_axon, theta1_axon, theta2_axon)
        
        dendrite_polys.append(dendrite_poly)
        axon_polys.append(axon_poly)
    
    return [axon_polys, dendrite_polys]


def generate_synapses(axon_polys: list, dendrite_polys: list) -> list:
    """
    Generates synapse connections based on overlaps between axonal and dendritic polygons.
    
    For each pair (axon from one neuron, dendrite from another), if there is an overlap,
    a connection object is created. Additionally, nested intersections are computed.
    
    :param axon_polys: List of axonal polygons.
    :param dendrite_polys: List of dendritic polygons.
    :return: List of connection objects representing synapses.
    """
    tmp_storage = []
    print(len(axon_polys), len(dendrite_polys))
    for i in range(len(axon_polys)):
        axon = axon_polys[i]
        for j in range(len(dendrite_polys)):
            if i == j:
                continue
            dendrite = dendrite_polys[j]
            overlap = axon.intersection(dendrite)
            if not overlap.is_empty:
                a_ij = overlap.area

                p_ij = 1 - np.exp(-alpha * a_ij)

                if np.random.random() < p_ij:
                    con = connection()
                    con.hosts = [i]
                    con.connection_poly = overlap
                    con.connections = [j]
                    tmp_storage.append(con)
                    # Recursively search for nested intersections.
                    nested_intersections = get_nested_intersections(con, dendrite_polys)
                    tmp_storage += nested_intersections
    return tmp_storage


def get_nested_intersections(intersection: connection, polys: list, poly_type: str = "dendrite") -> list:
    """
    Recursively finds nested intersections between an existing intersection polygon and a list of polygons.
    
    :param intersection: A connection object whose connection_poly is used for intersection.
    :param polys: List of polygons to check for further overlaps.
    :param poly_type: Either "dendrite" or "axon", determining which connection list to update.
    :return: A list of new connection objects for each nested intersection found.
    """
    new_intersections = []
    for i in range(len(polys)):
        # Skip indices already involved in the current connection.
        if i in intersection.hosts or i in intersection.connections:
            continue
        overlap = intersection.connection_poly.intersection(polys[i])
        if not overlap.is_empty:
            new_connection = intersection.copy()
            new_connection.connection_poly = overlap
            if poly_type == "dendrite":
                new_connection.add_connection(i)
            elif poly_type == "axon":
                new_connection.add_host(i)
            new_intersections.append(new_connection)
            new_intersections += get_nested_intersections(new_connection, polys, poly_type)
    return new_intersections


def find_overlap_points(x1, y1, x2, y2, threshold: float = 0.1):
    """
    Finds overlapping points between two sets of coordinates that are within a given threshold distance.
    
    :param x1: List of x-coordinates (first set).
    :param y1: List of y-coordinates (first set).
    :param x2: List of x-coordinates (second set).
    :param y2: List of y-coordinates (second set).
    :param threshold: Distance threshold for overlap.
    :return: Two lists: overlapping x-coordinates and overlapping y-coordinates.
    """
    overlap_x = []
    overlap_y = []
    for i in range(len(x1)):
        for j in range(len(x2)):
            if np.sqrt((x1[i] - x2[j])**2 + (y1[i] - y2[j])**2) < threshold:
                overlap_x.append((x1[i] + x2[j]) / 2)
                overlap_y.append((y1[i] + y2[j]) / 2)
    return overlap_x, overlap_y


def create_poly_params(soma_points: list) -> list:
    """
    Generates polygon parameters (radii and angular ranges) for axonal and dendritic regions
    based on neuron soma positions.
    
    :param soma_points: List of Shapely Points for neuron somata.
    :return: List [r_dendrite, r_axon, dendrite_thetas, axon_thetas].
    """
    axon_angle = np.pi / 4
    dendrite_angle = np.pi / 2.5
    r_axon = 2
    r_dendrite = 1

    axon_thetas = []
    dendrite_thetas = []
    overall_direction = 0
    direction_variance = np.pi / 3

    # Create random variance for each neuron.
    variances = (np.random.rand(len(soma_points)) * direction_variance) - (direction_variance / 2)
    axon_directions = overall_direction + variances

    for i in range(len(soma_points)):
        axon_direction = axon_directions[i]
        theta1_axon = axon_direction + (axon_angle / 2)
        theta2_axon = axon_direction - (axon_angle / 2)
        axon_thetas.append((theta1_axon, theta2_axon))
        
        dendrite_direction = np.pi + axon_direction
        theta1_dendrite = dendrite_direction + (dendrite_angle / 2)
        theta2_dendrite = dendrite_direction - (dendrite_angle / 2)
        dendrite_thetas.append((theta1_dendrite, theta2_dendrite))

    return [r_dendrite, r_axon, dendrite_thetas, axon_thetas]


def get_axon_overlap(synapses: list, axon_polys: list) -> list:
    """
    Identifies additional synaptic connections based on overlaps between axonal polygons and existing synapses.
    
    :param synapses: List of current connection objects.
    :param axon_polys: List of axonal polygons.
    :return: List of new connection objects representing additional overlaps.
    """
    new_synapses = []
    for synapse_connection in synapses:
        new_synapses += get_nested_intersections(synapse_connection, axon_polys, poly_type="axon")
    return remove_duplicate_intersections(new_synapses)


def create_synapses(soma_points: list) -> list:
    """
    Creates synapse connection objects based on neuron soma positions.
    
    Steps:
      1. Convert soma_points to Shapely Points if needed.
      2. Generate axonal and dendritic polygons.
      3. Generate synapses from dendritic overlaps.
      4. Identify additional synapses from axonal overlaps.
      5. Remove duplicate synapse connections.
    
    :param soma_points: List of neuron positions (as Points or tuples).
    :return: List of connection objects representing synapses.
    """
    global axon_polys
    global dendrite_polys

    # Convert tuples to Points if necessary.
    if isinstance(soma_points[0], tuple):
        soma_points = list(map(Point, soma_points))
    
    r_dendrite, r_axon, dendrite_thetas, axon_thetas = create_poly_params(soma_points)
    print("Creating polygons")
    axon_polys, dendrite_polys = generate_neuron_polys(soma_points, r_dendrite, r_axon, dendrite_thetas, axon_thetas)
    print("Generating synapses")
    no_overlap_synapses = generate_synapses(axon_polys, dendrite_polys)
    print("Removing any duplicate synapses")
    no_overlap_synapses = remove_duplicate_intersections(no_overlap_synapses)
    print("Getting axons that overlap existing synapses")
    synapses = get_axon_overlap(no_overlap_synapses, axon_polys)
    print("Found")
    synapses += no_overlap_synapses
    print("Total synapses:", len(synapses))
    return synapses


def remove_duplicate_intersections(connections: list) -> list:
    """
    Removes duplicate connection objects by comparing sorted host and connection lists.
    
    :param connections: List of connection objects.
    :return: List of unique connection objects.
    """
    unique_connections = {}
    for con in connections:
        sorted_hosts = sorted(con.hosts)
        sorted_connections = sorted(con.connections)
        key = f"{str(sorted_hosts)}, {str(sorted_connections)}"
        unique_connections[key] = con
    return list(unique_connections.values())


# ------------------- Main Plotting Section -------------------

if __name__ == '__main__':
    fig, ax_plot = plt.subplots(num="Probabilistic Synaptic Generator")
    max_size = 10
    num_neurons = 25

    graphing_colors = {"synapse": "#00bcd9"}
    fig.suptitle("Synaptic Generator")


    # Generate random neuron positions.
    soma_x = np.random.rand(num_neurons) * max_size
    soma_y = np.random.rand(num_neurons) * max_size
    soma_points = [Point(soma_x[i], soma_y[i]) for i in range(num_neurons)]
    
    # Generate synapses based on soma points.
    synapses = create_synapses(soma_points)

    # Plot soma points.
    for point_ind, point in enumerate(soma_points):
        plot_point(ax_plot, point, "#fc2803", annotation=str(point_ind))
    # Plot axonal polygons.
    for poly in axon_polys:
        plot_filled_polygon(ax_plot, poly, "#03fc6f")
    # Plot dendritic polygons.
    for poly in dendrite_polys:
        plot_filled_polygon(ax_plot, poly, "#f003fc")
    # Plot synapse connection polygons and synapse positions.
    for syn in synapses:
        print(syn)
        plot_filled_polygon(ax_plot, syn.connection_poly, "#1c1c1c")
        plot_point(ax_plot, Point(syn.get_center()), graphing_colors["synapse"], border_width=1)
    
    legend_elements = [
        Patch(facecolor="#03fc6f", label="Axons"),
        Patch(facecolor="#f003fc", label="Dendrites"),
        Patch(facecolor=graphing_colors["synapse"], label="Synapse"),
        Patch(facecolor="#1c1c1c", label="Synapse Area"),
        Patch(facecolor="#fc2803", label="Neuron")
    ]
    plt.subplots_adjust(right=0.75)
    ax_plot.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    ax_plot.set_aspect("equal")
    plt.draw()
    plt.show()
