import random
import json
import csv
import argparse
import multiprocessing as mp
import numpy as np
from shapely.geometry import shape, Point

# Try to import CUDA libraries, if available
try:
    import cupy as cp
    import cuspatial
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

def load_geojson(file_path):
    """Load and parse the GeoJSON file."""
    with open(file_path, 'r') as f:
        geojson = json.load(f)
    return geojson

def get_bounds(geojson):
    """Extract the bounding box of the GeoJSON geometry."""
    if 'bbox' in geojson:
        return geojson['bbox']
    
    # If bbox not provided, calculate it
    if 'type' in geojson and geojson['type'] == 'FeatureCollection':
        features = geojson['features']
        coords = []
        for feature in features:
            geometry = feature['geometry']
            if geometry['type'] == 'Polygon':
                for ring in geometry['coordinates']:
                    coords.extend(ring)
            elif geometry['type'] == 'MultiPolygon':
                for polygon in geometry['coordinates']:
                    for ring in polygon:
                        coords.extend(ring)
    elif 'type' in geojson and geojson['type'] == 'Feature':
        geometry = geojson['geometry']
        if geometry['type'] == 'Polygon':
            for ring in geometry['coordinates']:
                coords.extend(ring)
        elif geometry['type'] == 'MultiPolygon':
            for polygon in geometry['coordinates']:
                for ring in polygon:
                    coords.extend(ring)
    else:
        # Assume it's just a geometry
        if geojson['type'] == 'Polygon':
            for ring in geojson['coordinates']:
                coords.extend(ring)
        elif geojson['type'] == 'MultiPolygon':
            for polygon in geojson['coordinates']:
                for ring in polygon:
                    coords.extend(ring)
    
    lons = [coord[0] for coord in coords]
    lats = [coord[1] for coord in coords]
    return [min(lons), min(lats), max(lons), max(lats)]

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using Shapely."""
    return polygon.contains(Point(point))

def generate_point_in_polygon(polygon, bounds):
    """Generate a random point within the bounds and check if it's in the polygon."""
    min_lon, min_lat, max_lon, max_lat = bounds
    while True:
        lon = random.uniform(min_lon, max_lon)
        lat = random.uniform(min_lat, max_lat)
        if point_in_polygon((lon, lat), polygon):
            return lat, lon

def worker_process(args):
    """Worker function for multiprocessing."""
    polygon, bounds, num_points = args
    results = []
    for _ in range(num_points):
        lat, lon = generate_point_in_polygon(polygon, bounds)
        frequency = random.randint(1, 15)
        results.append((lat, lon, frequency))
    return results

def generate_with_multiprocessing(geojson_data, num_points, num_processes):
    """Generate points using multiprocessing."""
    # Get the polygon and bounds
    if 'type' in geojson_data and geojson_data['type'] == 'FeatureCollection':
        # Use the first feature for simplicity
        geometry = geojson_data['features'][0]['geometry']
    elif 'type' in geojson_data and geojson_data['type'] == 'Feature':
        geometry = geojson_data['geometry']
    else:
        # Assume it's just a geometry
        geometry = geojson_data
    
    polygon = shape(geometry)
    bounds = get_bounds(geojson_data)
    
    # Split work across processes
    points_per_process = max(1, num_points // num_processes)
    remainder = num_points % num_processes
    
    args_list = []
    for i in range(num_processes):
        points_to_generate = points_per_process + (1 if i < remainder else 0)
        if points_to_generate > 0:
            args_list.append((polygon, bounds, points_to_generate))
    
    # Use multiprocessing to generate points
    with mp.Pool(processes=num_processes) as pool:
        results_list = pool.map(worker_process, args_list)
    
    # Flatten results
    all_results = []
    for result in results_list:
        all_results.extend(result)
    
    return all_results

def generate_with_cuda(geojson_data, num_points):
    """Generate points using CUDA if available."""
    if not HAS_CUDA:
        raise ImportError("CUDA libraries not available")
    
    # Get the bounds
    bounds = get_bounds(geojson_data)
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Generate random points
    # Note: This is a simplified version - a real implementation would
    # need more complex CUDA kernels for point-in-polygon tests
    
    # First generate many more points than needed
    factor = 5  # Oversample factor
    rand_lons = cp.random.uniform(min_lon, max_lon, num_points * factor)
    rand_lats = cp.random.uniform(min_lat, max_lat, num_points * factor)
    
    # Convert GeoJSON to a format usable by cuspatial
    if 'type' in geojson_data and geojson_data['type'] == 'FeatureCollection':
        geometry = geojson_data['features'][0]['geometry']
    elif 'type' in geojson_data and geojson_data['type'] == 'Feature':
        geometry = geojson_data['geometry']
    else:
        geometry = geojson_data
    
    # Point-in-polygon test using cuspatial (simplified)
    # In a real implementation, we would convert the geometry to cuspatial format
    # and use cuspatial's point_in_polygon
    
    # For now, we'll fall back to CPU for the point-in-polygon test
    points = np.vstack((cp.asnumpy(rand_lons), cp.asnumpy(rand_lats))).T
    polygon = shape(geometry)
    
    results = []
    count = 0
    for lon, lat in points:
        if count >= num_points:
            break
        if point_in_polygon((lon, lat), polygon):
            frequency = random.randint(1, 15)
            results.append((lat, lon, frequency))
            count += 1
    
    return results

def save_to_csv(data, output_file):
    """Save the data to a CSV file."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['latitude', 'longitude', 'frequency'])
        writer.writerows(data)

def main():
    parser = argparse.ArgumentParser(description='Generate random points within a GeoJSON polygon')
    parser.add_argument('--geojson', required=True, help='Path to GeoJSON file')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    parser.add_argument('--num_points', type=int, default=1000, help='Number of points to generate')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--num_processes', type=int, default=mp.cpu_count(), 
                        help='Number of processes to use (default: number of CPU cores)')
    args = parser.parse_args()
    
    # Load the GeoJSON file
    geojson_data = load_geojson(args.geojson)
    
    # Generate points
    if args.use_cuda and HAS_CUDA:
        print(f"Generating {args.num_points} points using CUDA...")
        data = generate_with_cuda(geojson_data, args.num_points)
    else:
        if args.use_cuda and not HAS_CUDA:
            print("CUDA requested but not available. Falling back to multiprocessing.")
        print(f"Generating {args.num_points} points using {args.num_processes} processes...")
        data = generate_with_multiprocessing(geojson_data, args.num_points, args.num_processes)
    
    # Save the data to CSV
    save_to_csv(data, args.output)
    print(f"Generated {len(data)} points and saved to {args.output}")

if __name__ == "__main__":
    main()