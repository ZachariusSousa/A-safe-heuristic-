import osmnx as ox
import folium
from folium import plugins, Rectangle
import matplotlib
from matplotlib.colors import to_hex
import numpy as np
import pickle
import csv
import geopandas as gpd  # Import geopandas for GeoDataFrame
matplotlib.use("TkAgg")
from shapely.geometry import LineString
import networkx as nx


avg_risk = []

class Node:
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude


class Quadtree:
    def __init__(self, boundary, max_nodes):
        self.boundary = boundary  # (min_latitude, min_longitude, max_latitude, max_longitude)
        self.nodes = []  # Nodes contained in this node
        self.children = []  # Subdivisions of the quadtree
        self.max_nodes = max_nodes
        self.risk_value = None  # Risk value for this quadrant, representing its level
        self.max_depth = 0

    def find_quad_risk_for_coordinate(self, coordinate):
        if not self.in_boundary_from_coordinates(coordinate):
            return None

        if not self.children:
            if len(self.nodes) == 0:
                return self.risk_value / 2
            return self.risk_value + len(self.nodes)

        for child in self.children:
            quad_risk = child.find_quad_risk_for_coordinate(coordinate)
            if quad_risk is not None:
                return quad_risk

        return None

    def insert_node(self, node, current_level=0):
        if not self.in_boundary(node):
            return

        if len(self.nodes) < self.max_nodes:
            self.nodes.append(node)
        else:
            if not self.children:
                self.subdivide()

            for child in self.children:
                child.insert_node(node, current_level + 1)

        # After inserting a node, update the risk value to represent the current level
        self.update_risk(current_level)

    def update_risk(self, parent_risk):
        # Set the risk value of this quadrant based on the parent's risk value
        self.risk_value = parent_risk + 1
        if self.risk_value > self.max_depth:
            self.max_depth = self.risk_value

    def in_boundary_from_coordinates(self, coordinates):
        return (
            self.boundary[0] <= coordinates[1] <= self.boundary[2] and
            self.boundary[1] <= coordinates[0] <= self.boundary[3]
        )

    def in_boundary(self, node):
        return (
            self.boundary[0] <= node.latitude <= self.boundary[2] and
            self.boundary[1] <= node.longitude <= self.boundary[3]
        )

    def subdivide(self):
        x_mid = (self.boundary[0] + self.boundary[2]) / 2
        y_mid = (self.boundary[1] + self.boundary[3]) / 2

        # Create children quadrants
        child1 = Quadtree((self.boundary[0], self.boundary[1], x_mid, y_mid), self.max_nodes)
        child2 = Quadtree((x_mid, self.boundary[1], self.boundary[2], y_mid), self.max_nodes)
        child3 = Quadtree((self.boundary[0], y_mid, x_mid, self.boundary[3]), self.max_nodes)
        child4 = Quadtree((x_mid, y_mid, self.boundary[2], self.boundary[3]), self.max_nodes)

        # Distribute nodes to children
        for node in self.nodes:
            for child in [child1, child2, child3, child4]:
                child.insert_node(node)

        # Update children list and clear nodes from the current level
        self.children.extend([child1, child2, child3, child4])
        self.nodes.clear()

        # Update the risk value for each child based on the current level
        for child in self.children:
            child.update_risk(self.risk_value)

    def query_boundary(self, query_boundary, result_nodes=None):
        if result_nodes is None:
            result_nodes = []

        if not self.boundary_overlap(query_boundary):
            return result_nodes
        else:
            for node in self.nodes:
                if self.node_in_boundary(node, query_boundary):
                    result_nodes.append(node)

            for child in self.children:
                child.query_boundary(query_boundary, result_nodes)

        return result_nodes

    def boundary_overlap(self, query_boundary):
        if len(query_boundary) < 4 or len(self.boundary) < 4:
            return False

        return not (
                query_boundary[2] < self.boundary[0] or
                query_boundary[0] > self.boundary[2] or
                query_boundary[3] < self.boundary[1] or
                query_boundary[1] > self.boundary[3]
        )
    def node_in_boundary(self, node, query_boundary):
        return (
            query_boundary[0] <= node.latitude <= query_boundary[2] and
            query_boundary[1] <= node.longitude <= query_boundary[3]
        )

    def save_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)


def dms_to_decimal(degrees):
    d, t = divmod(degrees, 10000000)
    m, s = divmod(t, 100000)
    return d + (m / 60.0) + (s / 1000.0 / 3600.0)

def read_nodes_from_csv(file_path):
    nodes = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            try:
                latitude = dms_to_decimal(float(row[0]))
                longitude = dms_to_decimal(float(row[1]))
                nodes.append(Node(latitude, longitude))
            except (ValueError, IndexError):
                print(f"Skipping invalid row: {row}")
    return nodes


def plot_quadtree_folium(quadtree, m=None, depth=0):
    max_depth = 12
    max_nodes_per_quad = quadtree.max_nodes

    if m is None:
        # If m is not provided, create a new Folium map centered around the quadtree's boundary
        center_latitude = (quadtree.boundary[0] + quadtree.boundary[2]) / 2
        center_longitude = (quadtree.boundary[1] + quadtree.boundary[3]) / 2
        m = folium.Map(location=[center_latitude, center_longitude], zoom_start=14)

    # Color mapping based on the number of nodes and size of the quadrant
    color_map = 'hot_r'  # '_r' indicates reverse colormap

    # Calculate normalized factors for node_count and depth
    node_count = len(quadtree.nodes)
    node_count_factor = np.log1p(node_count) / np.log1p(max_nodes_per_quad) / 4

    # Depth factor normalization - adjust the scale as needed
    depth_factor = depth / max_depth * 1.3  # max_depth should be set to the maximum depth of your quadtree

    # Combine normalized factors
    combined_factor = (node_count_factor * 3 + depth_factor) / 3

    # Use the combined factor to get the color from the colormap
    color = matplotlib.cm.get_cmap(color_map)(combined_factor)

    # Convert the color to a hex string
    color_hex = to_hex(color)

    # Create a Rectangle to represent the boundary of the quadrant
    rectangle = Rectangle(
        bounds=[(quadtree.boundary[0], quadtree.boundary[1]),
                (quadtree.boundary[2], quadtree.boundary[3])],
        color=color_hex,
        fill=True,
        fill_color=color_hex,
        fill_opacity=0.1
    )
    rectangle.add_to(m)

    # Recursively plot the children
    for i, child in enumerate(quadtree.children):
        plot_quadtree_folium(child, m, depth=depth + 1)

    # Plot the nodes within the current node
    for node in quadtree.nodes:
        folium.CircleMarker(location=[node.latitude, node.longitude], radius=3, color='black').add_to(m)

    return m

def plot_road_graph(folium_map, road_graph):
    # Convert the networkx graph to a GeoDataFrame
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(road_graph)

    # Plot nodes
    for idx, row in gdf_nodes.iterrows():
        folium.CircleMarker(location=[row['y'], row['x']], radius=3, color='red').add_to(folium_map)

    # Plot edges
    folium.GeoJson(gdf_edges.geometry).add_to(folium_map)

    # Use the folium FastMarkerCluster plugin to cluster nodes for better performance
    plugins.FastMarkerCluster(data=list(zip(gdf_nodes['y'], gdf_nodes['x']))).add_to(folium_map)

    return folium_map

def plot_shortest_path(m, graph, start_point, end_point):

    start = ox.nearest_nodes(graph, start_point[1], start_point[0])
    end = ox.nearest_nodes(graph, end_point[1], end_point[0])

    #shortest_path = nx.astar_path(graph, start, end, heuristic=lambda n, g: custom_heuristic(n, g, graph))
    shortest_path = custom_shortest_path(graph, start_point, end_point, custom_weights)
    #shortest_path = calculate_shortest_path(graph, start_point, end_point)

    # Convert the networkx graph to a GeoDataFrame for shortest path visualization
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(graph)

    # Plot the edges of the shortest path using a boolean mask
    mask_shortest_path = gdf_edges.index.isin(list(zip(shortest_path, shortest_path[1:])))
    shortest_path_edges = gdf_edges[mask_shortest_path]

    # Create a GeoDataFrame for start and end points
    gdf_start_end = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([start_point[1], end_point[1]], [start_point[0], end_point[0]]),
        columns=['geometry'], crs=gdf_nodes.crs)

    # Plot the edges of the graph
    folium.GeoJson(shortest_path_edges.geometry).add_to(m)

    # Plot the nodes of the shortest path
    for idx, row in gdf_nodes.loc[shortest_path].iterrows():
        folium.CircleMarker(location=[row['y'], row['x']], radius=5, color='blue', fill=True).add_to(m)

    # Plot start and end points
    for idx, row in gdf_start_end.iterrows():
        folium.Marker(location=[row.geometry.y, row.geometry.x], popup=row.geometry,
                      icon=folium.Icon(color='green')).add_to(m)

    return m

def calculate_shortest_path(graph, start_point, end_point):
    w = "length"  # Use "length" as the weight attribute
    node_one = ox.nearest_nodes(graph, start_point[1], start_point[0])
    node_two = ox.nearest_nodes(graph, end_point[1], end_point[0])
    shortest_path = ox.shortest_path(graph, node_one, node_two, weight=w)
    return shortest_path


def assign_risks(graph, quadtree):
    for edge in graph.edges(data=True):
        edge_data = edge[2]  # Edge attributes dictionary

        # Calculate risk values for the edge
        risk_values = calculate_risk_for_edge_based_on_quadtree(edge_data, quadtree)

        # Assign the list of risk values to the edge data
        edge_data["risk"] = aggregate_risk_values(risk_values)


def aggregate_risk_values(risk_values):
    return sum(risk_values) / len(risk_values) if risk_values else 0



def calculate_risk_for_edge_based_on_quadtree(edge_data, quadtree, num_interpolation_points=10):
    edge_coords_list = get_edge_coordinates(edge_data)

    # Initialize a list to store individual risk values for each point
    risk_values = []

    if not edge_coords_list:
        return risk_values

    # Interpolate points along the edge
    interpolated_points = interpolate_points(edge_coords_list, num_interpolation_points)

    # Check the risk for each interpolated point
    for point in interpolated_points:
        risk_value = quadtree.find_quad_risk_for_coordinate(point)
        if risk_value is not None:
            risk_values.append(risk_value)

    return risk_values


def interpolate_points(edge_coords, num_points):
    interpolated_points = []

    # Interpolate additional points between consecutive coordinates
    for i in range(len(edge_coords) - 1):
        lat1, lon1 = edge_coords[i]
        lat2, lon2 = edge_coords[i + 1]

        for j in range(num_points):
            alpha = j / num_points
            interpolated_lat = lat1 + alpha * (lat2 - lat1)
            interpolated_lon = lon1 + alpha * (lon2 - lon1)
            interpolated_points.append((interpolated_lat, interpolated_lon))

    return interpolated_points


def get_edge_coordinates(edge_data):

    geometry_data = edge_data.get('geometry')
    if geometry_data:
        try:
            if isinstance(geometry_data, LineString):
                # Extract coordinates from LineString
                coordinates_list = list(geometry_data.coords)
                return coordinates_list
            else:
                # Convert string representation to LineString and extract coordinates
                line = LineString(map(float, pair.split()) for pair in str(geometry_data).split(', '))

                # Return a list of tuples representing each point on the geometry
                return list(line.coords)
        except (IndexError, ValueError):
            pass

    # Return an empty list if geometry_data is None or extraction fails
    return []

def calculate_risk_from_intersected_quads(intersected_quads):
    # Replace this with your logic to calculate risk based on intersected quads
    # For example, you might average the risk values of the intersected quads

    if intersected_quads:
        average_risk = sum(quad.risk_value for quad in intersected_quads) / len(intersected_quads)
        return average_risk
    else:
        return 0.5  # Default risk if no intersection


def custom_weights(edge_attributes, distance_weight=1, risk_weight=0):

    average = aggregate_risk_values(avg_risk)

    # Extract edge length (distance)
    length = edge_attributes[0].get("length", 1)

    # Extract edge risk value
    risk_value = edge_attributes[0].get("risk", 1)

    if risk_value == 0:
        risk_value = 0

    # Calculate combined weight based on distance and risk
    combined_weight = (distance_weight * length) + (risk_weight * risk_value)

    return combined_weight


def custom_shortest_path(graph, start_point, end_point, weights):
    start = ox.nearest_nodes(graph, start_point[1], start_point[0])
    end = ox.nearest_nodes(graph, end_point[1], end_point[0])

    # Initialize the distance dictionary
    distance = {node: float('inf') for node in graph.nodes}
    distance[start] = 0

    # Initialize the predecessors dictionary
    predecessors = {node: None for node in graph.nodes}

    # Priority queue to store nodes and their distances
    priority_queue = list(graph.nodes)

    while priority_queue:
        # Get the node with the smallest distance
        current_node = min(priority_queue, key=lambda node: distance[node])
        priority_queue.remove(current_node)

        # Break if the end node is reached
        if current_node == end:
            break

        # Update distances and predecessors for neighboring nodes
        for neighbor in graph.neighbors(current_node):
            # Calculate the edge weight
            edge_weight = weights(graph[current_node][neighbor])

            # Calculate the new distance
            new_distance = distance[current_node] + edge_weight

            print(distance[neighbor])
            # Check if the new distance is smaller than the current distance
            if new_distance < distance[neighbor]:
                distance[neighbor] = new_distance
                predecessors[neighbor] = current_node

    # Reconstruct the path
    path = []
    current_node = end
    while current_node is not None:
        path.insert(0, current_node)
        current_node = predecessors[current_node]

    return path

def calculate_risk_for_edge_based_on_d7(edge_data, graph):
    source_id = edge_data[0]
    target_id = edge_data[1]

    source_accidents = get_num_accidents(graph, source_id)
    target_accidents = get_num_accidents(graph, target_id)

    # Calculate the average number of accidents for the edge
    average_accidents = (source_accidents + target_accidents)

    return average_accidents


def get_num_accidents(graph, node_id):
    node_attributes = graph.nodes[node_id]
    num_accidents_str = node_attributes.get('num_accidents', '0')
    return int(num_accidents_str)

def assign_risks_Astar(graph):
    for edge in graph.edges(data=True):
        source_id, target_id = edge[:2]  # Extracting source and target node IDs
        # Calculate risk values for the edge based on d7 values of source and target nodes
        risk_value = calculate_risk_for_edge_based_on_d7((source_id, target_id), graph)
        # Assign the risk value to the edge data
        edge[2]["risk"] = risk_value

def custom_heuristic(node, goal, graph):
    # Custom heuristic: consider the risk value as the distance estimate
    return graph.nodes[node].get("risk", 0)


def custom_shortest_path_Astar(graph, start_point, end_point):
    # Convert node coordinates to numeric values
    for node in graph.nodes(data=True):
        node[1]['x'] = float(node[1]['x'])
        node[1]['y'] = float(node[1]['y'])

    start = ox.nearest_nodes(graph, start_point[1], start_point[0])
    end = ox.nearest_nodes(graph, end_point[1], end_point[0])

    # A* algorithm with the custom heuristic
    path = nx.astar_path(graph, start, end, heuristic=lambda n, g: custom_heuristic(n, g, graph), weight="risk")

    return path


def plot_risk_path_Astar(m, graph, start_point, end_point):
    shortest_path = custom_shortest_path_Astar(graph, start_point, end_point)
    # Convert the networkx graph to a GeoDataFrame for shortest path visualization
    gdf_nodes = ox.graph_to_gdfs(graph, nodes=True, edges=False)

    # Create a list of LineString geometries for the edges in the shortest path
    edge_geometries = []
    for i in range(len(shortest_path) - 1):
        edge_data = graph[shortest_path[i]][shortest_path[i + 1]]
        if 'geometry' in edge_data:
            edge_geometry = edge_data['geometry']
        else:
            edge_geometry = LineString([(graph.nodes[shortest_path[i]]['x'], graph.nodes[shortest_path[i]]['y']),
                                        (graph.nodes[shortest_path[i + 1]]['x'], graph.nodes[shortest_path[i + 1]]['y'])])
        edge_geometries.append(edge_geometry)

    # Create a GeoDataFrame for edges
    edges_data = {'u': shortest_path[:-1], 'v': shortest_path[1:], 'geometry': edge_geometries}
    gdf_edges = gpd.GeoDataFrame(edges_data, columns=['u', 'v', 'geometry'], crs=gdf_nodes.crs)


    # Ensure the 'geometry' field contains valid Shapely geometries
    gdf_edges['geometry'] = gdf_edges['geometry'].apply(lambda x: LineString(x.coords) if isinstance(x, LineString) else x)

    # Plot the edges of the shortest path using a boolean mask
    mask_shortest_path = gdf_edges.index.isin(range(len(shortest_path) - 1))
    shortest_path_edges = gdf_edges[mask_shortest_path]

    # Create a GeoDataFrame for start and end points
    gdf_start_end = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([start_point[1], end_point[1]], [start_point[0], end_point[0]]),
        columns=['geometry'], crs=gdf_nodes.crs)

    # Plot the edges of the graph
    folium.GeoJson(shortest_path_edges.geometry).add_to(m)

    # Plot the nodes of the shortest path
    for idx, row in gdf_nodes.loc[shortest_path].iterrows():
        folium.CircleMarker(location=[row['y'], row['x']], radius=5, color='blue', fill=True).add_to(m)

    # Plot start and end points
    for idx, row in gdf_start_end.iterrows():
        folium.Marker(location=[row.geometry.y, row.geometry.x], popup=row.geometry,
                      icon=folium.Icon(color='green')).add_to(m)

    return m


