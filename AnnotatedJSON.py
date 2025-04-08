"""
Optimized Route Analysis System

This module provides tools for analyzing transportation routes using network analysis.
It loads road networks, calculates optimal paths between points, and finds alternative routes.

Key Features:
- Loads road networks from GeoPackage files
- Processes origin-destination pairs
- Calculates shortest paths using network analysis
- Identifies alternative routes
- Exports results in multiple formats (GeoPackage, CSV)
"""

import os
import time
import logging
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, MultiLineString
from pyproj import Transformer
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

# Configure logging to track processing events and errors
logging.basicConfig(
    filename='route_analysis.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OptimizedRouteAnalyzer:
    """
    Main class for route analysis operations.
    
    Handles network loading, route calculation, and result export.
    Uses EPSG:3005 (BC Albers) for spatial calculations and EPSG:4326 (WGS84) for lat/lon output.
    """
    
    def __init__(self):
        """Initialize the route analyzer with default settings and coordinate transformers"""
        # Coordinate transformers for BC Albers <-> WGS84 conversion
        self.transformer_3005_to_4326 = Transformer.from_crs('EPSG:3005', 'EPSG:4326', always_xy=True)
        self.transformer_4326_to_3005 = Transformer.from_crs('EPSG:4326', 'EPSG:3005', always_xy=True)
        
        # Bounding box for British Columbia (in EPSG:3005 coordinates)
        self.bounds = {
            'x_min': 1000000,   # Western BC boundary
            'x_max': 2000000,   # Eastern BC boundary
            'y_min': 400000,    # Southern BC boundary
            'y_max': 2000000    # Northern BC boundary
        }
        
        # Snap distances in meters for connecting points to network (increasing order)
        self.snap_distances = [250, 500, 1000, 2000]  # Increased maximum snap distance
        
        # Maximum attempts for route calculation retries
        self.max_attempts = 3
        
        # Output path for CSV results
        self.csv_output_path = r"C:\Users\caoim\OneDrive\Documents\Tranportation\DetourTestCSV.csv"
        
        # Minimum valid route distance (avoid zero-length routes)
        self.min_route_distance = 10  # Minimum valid route distance in meters

    def validate_coordinates(self, coord: Tuple[float, float]) -> bool:
        """
        Validate that coordinates are within expected bounds and properly formatted.
        
        Args:
            coord: Tuple of (x, y) coordinates in EPSG:3005
            
        Returns:
            bool: True if coordinates are valid and within bounds
        """
        try:
            x, y = coord[:2]
            # Check if coordinates fall within BC bounds
            if not (self.bounds['x_min'] <= x <= self.bounds['x_max'] and
                   self.bounds['y_min'] <= y <= self.bounds['y_max']):
                return False
            return True
        except (TypeError, ValueError, IndexError):
            # Catch cases where input isn't a proper coordinate pair
            return False

    def load_network(self, network_gpkg: str) -> nx.DiGraph:
        """
        Load and validate a road network from a GeoPackage file.
        
        Args:
            network_gpkg: Path to GeoPackage file containing road network
            
        Returns:
            nx.DiGraph: Directed graph representing the road network
            
        Raises:
            Exception: If network loading fails
        """
        try:
            print("\nLoading and validating network...")
            start = time.time()
            
            # Read the network data and ensure it's in EPSG:3005
            gdf = gpd.read_file(network_gpkg)
            if gdf.crs.to_epsg() != 3005:
                gdf = gdf.to_crs('EPSG:3005')
            
            # Initialize directed graph
            G = nx.DiGraph()
            valid_edges = []
            
            # Process each road segment in the network
            for _, row in gdf.iterrows():
                geom = row.geometry
                
                # Skip invalid or empty geometries
                if geom.is_empty or not geom.is_valid:
                    continue
                    
                # Handle both LineString and MultiLineString geometries
                lines = geom.geoms if geom.geom_type == 'MultiLineString' else [geom]
                
                for line in lines:
                    coords = list(line.coords)
                    if len(coords) >= 2:  # Need at least 2 points to form a line
                        # Validate all coordinates in the line
                        if not all(self.validate_coordinates(coord) for coord in coords):
                            continue
                            
                        # Calculate edge weight based on length and speed limit
                        speed = max(float(row.get('SPEED_LIMIT', 50)), 1)  # Ensure speed > 0
                        valid_edges.append((
                            coords[0],  # Start node
                            coords[-1],  # End node
                            {
                                'geometry': line,
                                'length': line.length,
                                'speed': speed,
                                'weight': line.length / speed  # Travel time as weight
                            }
                        ))
            
            # Add all valid edges to the graph
            G.add_edges_from(valid_edges)
            
            # Add reverse edges for bidirectional travel (unless marked as closed)
            reverse_edges = [(v, u, d) for u, v, d in valid_edges 
                           if str(d.get('is_closed', 'false')).lower() != 'true']
            G.add_edges_from(reverse_edges)
            
            print(f"Network loaded in {time.time()-start:.1f}s | Nodes: {len(G.nodes())} | Edges: {len(G.edges())}")
            return G
            
        except Exception as e:
            logging.exception("Network loading failed")
            raise

    def load_routes(self, routes_gpkg: str) -> List[Dict]:
        """
        Load origin-destination pairs from a GeoPackage file.
        
        Args:
            routes_gpkg: Path to GeoPackage containing route points
            
        Returns:
            List[Dict]: List of route dictionaries with source/destination info
            
        Raises:
            Exception: If route loading fails
        """
        try:
            print("\nLoading and validating route points...")
            gdf = gpd.read_file(routes_gpkg)
            
            # Ensure routes are in EPSG:3005
            if gdf.crs.to_epsg() != 3005:
                gdf = gdf.to_crs('EPSG:3005')
            
            routes = []
            skipped = 0
            
            # Process points in pairs (origin, destination)
            for i in range(0, len(gdf) - 1, 2):
                origin = gdf.iloc[i].geometry
                dest = gdf.iloc[i+1].geometry
                
                # Validate geometry types (must be Points)
                if not (isinstance(origin, Point) and isinstance(dest, Point)):
                    skipped += 1
                    continue
                
                # Validate coordinate locations
                if not (self.validate_coordinates((origin.x, origin.y)) and \
                    self.validate_coordinates((dest.x, dest.y))):
                    skipped += 1
                    continue
                
                # Check for duplicate/overlapping points
                if origin.distance(dest) < self.min_route_distance:
                    skipped += 1
                    continue
                
                # Convert to lat/lon for output
                origin_lon, origin_lat = self.transformer_3005_to_4326.transform(origin.x, origin.y)
                dest_lon, dest_lat = self.transformer_3005_to_4326.transform(dest.x, dest.y)
                
                # Create route dictionary
                routes.append({
                    'source': (origin.x, origin.y),  # EPSG:3005 coords
                    'destination': (dest.x, dest.y),  # EPSG:3005 coords
                    'source_lat': origin_lat,  # WGS84 latitude
                    'source_lon': origin_lon,  # WGS84 longitude
                    'dest_lat': dest_lat,     # WGS84 latitude
                    'dest_lon': dest_lon,      # WGS84 longitude
                    'original_geometry': LineString([origin, dest]),  # Straight-line geometry
                    'original_length': origin.distance(dest),  # Straight-line distance
                    'attributes': {k: v for k, v in gdf.iloc[i].items() if k != 'geometry'},  # Original attributes
                    'original_index': i  # Reference to original data position
                })
            
            print(f"Created {len(routes)} valid route pairs (skipped {skipped} invalid pairs)")
            return routes
            
        except Exception as e:
            logging.exception("Route loading failed")
            raise

    def snap_to_network(self, G: nx.DiGraph, point: Tuple[float, float], max_distance: float) -> Optional[Tuple[float, float]]:
        """
        Snap a point to the nearest network node within specified distance.
        
        Args:
            G: Road network graph
            point: (x,y) coordinates to snap
            max_distance: Maximum snapping distance in meters
            
        Returns:
            Optional[Tuple]: Nearest network node coordinates, or None if none found
        """
        if not self.validate_coordinates(point):
            return None
            
        point_shp = Point(point)
        nearest_node = None
        min_dist = float('inf')
        
        # Find nearest node in graph
        for node in G.nodes():
            dist = point_shp.distance(Point(node))
            if dist < min_dist and dist <= max_distance:
                min_dist = dist
                nearest_node = node
                if dist < 5.0:  # Early exit for very close nodes (optimization)
                    break
                    
        return nearest_node if min_dist <= max_distance else None

    def reconstruct_path_geometry(self, G: nx.DiGraph, path: List[Tuple[float, float]]) -> LineString:
        """
        Reconstruct the complete geometry of a path through the network.
        
        Args:
            G: Road network graph
            path: List of node coordinates representing the path
            
        Returns:
            LineString or MultiLineString: Geometry of the complete path
        """
        try:
            if len(path) < 2:  # Need at least 2 points to form a path
                return None
                
            lines = []
            # Connect each pair of consecutive nodes
            for u, v in zip(path[:-1], path[1:]):
                if G.has_edge(u, v):
                    # Use the original geometry if available
                    lines.append(G[u][v]['geometry'])
                else:
                    # Fallback to straight line between nodes
                    lines.append(LineString([u, v]))
            
            # Return as single LineString or MultiLineString
            return MultiLineString(lines) if len(lines) > 1 else lines[0]
            
        except Exception as e:
            logging.error(f"Path reconstruction failed: {str(e)}")
            return None

    def calculate_routes(self, G: nx.DiGraph, routes: List[Dict]) -> List[Dict]:
        """
        Calculate shortest paths and alternative routes for all origin-destination pairs.
        
        Args:
            G: Road network graph
            routes: List of route dictionaries from load_routes()
            
        Returns:
            List[Dict]: Results with path geometries and metrics
        """
        results = []
        start = time.time()
        
        print("\nCalculating routes with improved snapping...")
        with tqdm(total=len(routes), desc="Processing routes") as pbar:
            for route in routes:
                # Initialize result dictionary with basic info
                result = {
                    'original_index': route['original_index'],
                    'source_lat': route['source_lat'],
                    'source_lon': route['source_lon'],
                    'dest_lat': route['dest_lat'],
                    'dest_lon': route['dest_lon'],
                    'direct_distance': route['original_length']  # Straight-line distance
                }
                
                try:
                    best_snap = None
                    best_dist = float('inf')
                    
                    # Try multiple snap distances to find best connection points
                    for snap_dist in self.snap_distances:
                        source = self.snap_to_network(G, route['source'], snap_dist)
                        dest = self.snap_to_network(G, route['destination'], snap_dist)
                        
                        if source and dest:
                            # Calculate total snap distance (source + destination)
                            dist = Point(route['source']).distance(Point(source)) + \
                                   Point(route['destination']).distance(Point(dest))
                            if dist < best_dist:
                                best_dist = dist
                                best_snap = (source, dest, snap_dist)
                    
                    if best_snap:
                        source, dest, snap_dist = best_snap
                        try:
                            # Calculate shortest path using travel time as weight
                            path = nx.shortest_path(G, source, dest, weight='weight')
                            geom = self.reconstruct_path_geometry(G, path)
                            
                            if geom:
                                result.update({
                                    'status': 'success',
                                    'shortest_geometry': geom,
                                    'shortest_length': geom.length,
                                    'snap_distance_used': snap_dist
                                })
                                
                                # Find alternative path (if available)
                                alt_path = self.find_alternative_path(G, path)
                                if alt_path:
                                    alt_geom = self.reconstruct_path_geometry(G, alt_path)
                                    if alt_geom:
                                        result.update({
                                            'alternative_geometry': alt_geom,
                                            'alternative_length': alt_geom.length,
                                            'detour_ratio': alt_geom.length / geom.length
                                        })
                        except (nx.NetworkXNoPath, nx.NodeNotFound):
                            result['status'] = 'no_path'
                    else:
                        result['status'] = 'snap_failed'
                        
                except Exception as e:
                    result['status'] = 'error'
                    logging.error(f"Route failed: {str(e)}")
                
                results.append(result)
                pbar.update(1)
        
        print(f"\nRouting completed in {time.time()-start:.1f}s")
        self.analyze_results(results)
        return results

    def find_alternative_path(self, G: nx.DiGraph, path: List[Tuple[float, float]]) -> Optional[List[Tuple[float, float]]]:
        """
        Find an alternative path by strategically removing edges from the original path.
        
        Args:
            G: Road network graph
            path: List of nodes representing the original path
            
        Returns:
            Optional[List]: Alternative path if found, None otherwise
        """
        temp_G = G.copy()
        
        # Remove segments from the original path (up to 3 edges)
        for i in range(min(3, len(path)-1)):
            u, v = path[i], path[i+1]
            if temp_G.has_edge(u, v):
                temp_G.remove_edge(u, v)
            if temp_G.has_edge(v, u):
                temp_G.remove_edge(v, u)
        
        try:
            # Find new shortest path with some edges removed
            return nx.shortest_path(temp_G, path[0], path[-1], weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def analyze_results(self, results: List[Dict]):
        """
        Analyze and display statistics about the routing results.
        
        Args:
            results: List of result dictionaries from calculate_routes()
        """
        stats = {
            'success': 0,
            'success_with_alt': 0,
            'snap_failed': 0,
            'no_path': 0,
            'error': 0
        }
        
        # Count different outcome types
        for res in results:
            status = res.get('status', 'error')
            stats[status] += 1
            if status == 'success' and 'alternative_geometry' in res:
                stats['success_with_alt'] += 1
        
        # Print summary statistics
        print("\n===== Analysis Results =====")
        print(f"Total valid routes: {len(results)}")
        print(f"Successfully routed: {stats['success']} ({stats['success']/len(results):.1%})")
        print(f"With alternatives: {stats['success_with_alt']}")
        print(f"Snap failed: {stats['snap_failed']} (try increasing snap distances)")
        print(f"No path found: {stats['no_path']}")
        print(f"Errors: {stats['error']}")

    def export_results(self, results: List[Dict], output_dir: str):
        """
        Export routing results to GeoPackage and CSV files.
        
        Args:
            results: List of result dictionaries
            output_dir: Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data containers with default values
        shortest_records = []
        alt_records = []
        csv_data = []
        
        for r in results:
            try:
                # Initialize CSV row with default values
                csv_row = {
                    'source_lat': r.get('source_lat', float('nan')),
                    'source_lon': r.get('source_lon', float('nan')),
                    'dest_lat': r.get('dest_lat', float('nan')),
                    'dest_lon': r.get('dest_lon', float('nan')),
                    'direct_distance_km': round(r.get('direct_distance', 0) / 1000, 3),
                    'direct_distance_m': int(r.get('direct_distance', 0)),
                    'status': r.get('status', 'error')  # Default status if missing
                }
                
                # Add routing results if available
                if 'shortest_length' in r:
                    csv_row.update({
                        'shortest_distance_km': round(r['shortest_length'] / 1000, 3),
                        'shortest_distance_m': int(r['shortest_length']),
                        'snap_distance_used': r.get('snap_distance_used', float('nan')),
                        'routing_efficiency': round(r.get('direct_distance', 0) / r['shortest_length'], 3) 
                        if r['shortest_length'] > 0 else float('nan')
                    })
                    
                    # Add alternative path info if available
                    if 'alternative_length' in r:
                        csv_row.update({
                            'alternative_distance_km': round(r['alternative_length'] / 1000, 3),
                            'alternative_distance_m': int(r['alternative_length']),
                            'distance_diff_km': round((r['alternative_length'] - r['shortest_length']) / 1000, 3),
                            'distance_diff_m': int(r['alternative_length'] - r['shortest_length']),
                            'detour_percentage': round(((r['alternative_length'] - r['shortest_length']) / r['shortest_length']) * 100, 1),
                            'detour_ratio': round(r.get('detour_ratio', 0), 3)
                        })
                
                csv_data.append(csv_row)
                
                # Prepare spatial outputs only for successful routes
                if r.get('status') == 'success' and 'shortest_geometry' in r:
                    shortest_records.append({
                        'geometry': r['shortest_geometry'],
                        **{k: v for k, v in csv_row.items() 
                        if k not in ['direct_distance_km', 'direct_distance_m']}
                    })
                    
                    if 'alternative_geometry' in r:
                        alt_records.append({
                            'geometry': r['alternative_geometry'],
                            **{k: v for k, v in csv_row.items() 
                            if k not in ['direct_distance_km', 'direct_distance_m']}
                        })
                        
            except Exception as e:
                logging.error(f"Error processing route {r.get('original_index', 'unknown')}: {str(e)}")
                continue
        
        # Export spatial data
        if shortest_records:
            gdf_shortest = gpd.GeoDataFrame(shortest_records, crs='EPSG:3005')
            gdf_shortest.to_file(os.path.join(output_dir, 'shortest_routes.gpkg'), driver='GPKG')
            print(f"\nExported {len(gdf_shortest)} shortest routes")
        
        if alt_records:
            gdf_alt = gpd.GeoDataFrame(alt_records, crs='EPSG:3005')
            gdf_alt.to_file(os.path.join(output_dir, 'alternative_routes.gpkg'), driver='GPKG')
            print(f"Exported {len(gdf_alt)} alternative routes")
        
        # Export CSV with cleaning
        if csv_data:
            df = pd.DataFrame(csv_data)
            
            # Clean and sort data
            df = df.dropna(subset=['source_lat', 'source_lon', 'dest_lat', 'dest_lon'], how='all')
            df['path_deviation'] = df['shortest_distance_km'] - df['direct_distance_km']
            
            # Select and order columns
            columns = [
                'source_lat', 'source_lon', 'dest_lat', 'dest_lon',
                'direct_distance_km', 'direct_distance_m',
                'shortest_distance_km', 'shortest_distance_m',
                'path_deviation', 'routing_efficiency',
                'alternative_distance_km', 'alternative_distance_m',
                'distance_diff_km', 'distance_diff_m',
                'detour_percentage', 'detour_ratio',
                'snap_distance_used', 'status'
            ]
            df = df[[col for col in columns if col in df.columns]]
            
            # Save CSV
            os.makedirs(os.path.dirname(self.csv_output_path), exist_ok=True)
            df.to_csv(self.csv_output_path, index=False)
            print(f"\nExported {len(df)} route metrics to {self.csv_output_path}")
            print("\nSample of exported CSV data:")
            print(df.head().to_string(index=False))

if __name__ == "__main__":
    """
    Main execution block for route analysis.
    
    Typical usage:
    1. Initialize analyzer
    2. Load network data
    3. Load route points
    4. Calculate routes
    5. Export results
    """
    try:
        # Initialize the route analyzer
        analyzer = OptimizedRouteAnalyzer()
        
        # Configure input/output paths
        INPUT_NETWORK = r"C:\Users\caoim\OneDrive\Documents\Tranportation\DRA\DRA\DRANetwork_StartEnd_SpeedLimit.gpkg"
        INPUT_ROUTES = r"C:\Users\caoim\OneDrive\Documents\Tranportation\DRA\DRA\DRA_YES_Speed_LimitStart_EndVertices.gpkg"
        OUTPUT_DIR = r"C:\Users\caoim\OneDrive\Documents\Tranportation\output_final"
        
        print("===== Optimized Route Analysis =====")
        
        # Clear previous outputs if they exist
        if os.path.exists(OUTPUT_DIR):
            for file in os.listdir(OUTPUT_DIR):
                os.remove(os.path.join(OUTPUT_DIR, file))
            os.rmdir(OUTPUT_DIR)
        
        # Load data and process routes
        network = analyzer.load_network(INPUT_NETWORK)
        routes = analyzer.load_routes(INPUT_ROUTES)
        
        if routes:
            results = analyzer.calculate_routes(network, routes)
            analyzer.export_results(results, OUTPUT_DIR)
        
        print("\n===== Processing Complete =====")
        
    except Exception as e:
        print(f"\n!!! ERROR: {str(e)}")
        logging.exception("Processing failed")