# main_controller.py

import osmnx as ox

from QuadTree import Node, Quadtree, plot_quadtree_folium, read_nodes_from_csv, plot_shortest_path, assign_risks, plot_risk_path_Astar, assign_risks_Astar

def main():
    # Save OSMnx graph
    north, south, east, west = 35.75, 35.7, 139.8, 139.75
    graph = ox.graph_from_bbox(north, south, east, west, network_type="drive")
    ox.save_graphml(graph, filepath='tokyo_graph.graphml')

    # Load OSMnx graph
    #graph = ox.load_graphml(filepath='modified_graph.graphml')
    road_graph = ox.project_graph(graph)

    # Plot Quadtree on Folium map
    # boundary = (30, 128, 45, 150)  #Japan boundary
    # boundary = (35.6, 139.5, 35.8, 139.9) #Tokyo boundary
    boundary = (35.7, 139.75, 35.75, 139.8)  # Tokyo boundary

    start_point = (35.745, 139.75)
    end_point = (35.701, 139.8)

    # List of CSV file paths
    csv_file_paths = ['honhyo_2019.csv', 'honhyo_2020.csv', 'honhyo_2021.csv', 'honhyo_2022.csv']

    # Create the top-level quadtree for Tokyo
    quadtree = Quadtree(boundary, 1)

    # Loop through each CSV file
    for csv_file_path in csv_file_paths:
        # Read nodes from the CSV file
        print("Reading " + csv_file_path + " now")
        csv_nodes = read_nodes_from_csv(csv_file_path)

        # Insert each node into the quadtree
        for node in csv_nodes:
            if node.latitude == 0:
                continue  # Skip nodes with latitude 0
            quadtree.insert_node(node)

    result_nodes = quadtree.query_boundary(boundary)

    assign_risks(graph, quadtree)

    # Assign risks based on d7 values
    assign_risks_Astar(graph)

    print("Total Accidents: " + str(len(result_nodes)))

    print("Started Plotting")

    folium_map = plot_quadtree_folium(quadtree)
    #folium_map = plot_shortest_path(folium_map, graph, start_point, end_point)
    folium_map = plot_risk_path_Astar(folium_map, graph, start_point, end_point)

    # Save the map to an HTML file
    folium_map.save('route_v3.html')

    # Save the updated quadtree to a file
    quadtree.save_to_file('quadtree_with_path.pkl')


if __name__ == "__main__":
    main()
