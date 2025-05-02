# GPU-Accelerated Optimal Warehouse Location Finder

A high-performance tool for finding the optimal e-commerce warehouse location based on customer delivery data using GPU acceleration.

## Overview

This repository contains code to determine the most cost-effective location for an e-commerce warehouse by analyzing customer delivery locations and order frequencies. The implementation leverages GPU acceleration via RAPIDS (cuDF, cuML, cuGraph) to handle large-scale data efficiently.

The optimization follows a two-stage approach:
1. **Global Optimization**: Find a global optimum using Haversine distance
2. **Local Refinement**: Refine the search within a 5km radius using A* algorithm on the road network

## Features

- **GPU-Accelerated Performance**: Utilize NVIDIA GPUs for fast computation on large datasets
- **Real-world Network Analysis**: Use actual road networks rather than just straight-line distances
- **Weighted Optimization**: Account for delivery frequency to each location
- **Interactive Visualization**: Plot the optimal location on city maps
- **Efficient Point-to-Network Mapping**: GPU-accelerated KNN for mapping delivery points to road network

## Requirements

- NVIDIA GPU with CUDA support
- RAPIDS suite (cuDF, cuPy, cuGraph, cuML)
- OSMnx
- Pandas
- NumPy
- Matplotlib
- tqdm

## Installation

```bash
# Create a conda environment with RAPIDS
conda create -n warehouse-opt -c rapidsai -c nvidia -c conda-forge \
    cudf=23.04 cuml=23.04 cugraph=23.04 python=3.9 cudatoolkit=11.8

# Activate the environment
conda activate warehouse-opt

# Install additional dependencies
pip install osmnx matplotlib tqdm

# Clone the repository
git clone https://github.com/your-username/optimal-warehouse-location.git
cd optimal-warehouse-location
```

## Usage

### Dataset

This project uses the [E-commerce User Data for Optimisation](https://www.kaggle.com/datasets/alpha000x/ecommerce-user-data-for-optimisation) dataset from Kaggle, which contains delivery location data for optimization purposes.

### Input Format

The application expects a CSV file with the following structure:

| latitude | longitude | frequency |
|----------|-----------|-----------|
| 22.4971  | 88.3114   | 2         |
| 22.5219  | 88.3499   | 1         |
| 22.6251  | 88.4036   | 10        |

Where:
- `latitude` and `longitude`: Geographic coordinates of delivery locations
- `frequency`: Number of deliveries to that location

### Running the Optimization

```python
# Basic usage
python main.py

# With custom parameters
python main.py --csv-file path/to/data.csv --place "City, Country" --radius 3000
```

## How It Works

1. **Load Network**: Download or load cached road network data from OpenStreetMap
2. **Load Points**: Import customer delivery locations from CSV
3. **Snap to Network**: Map each delivery point to the nearest node in the road network using GPU-accelerated KNN
4. **Global Optimization**: Find the node that minimizes total Haversine distance weighted by delivery frequency
5. **Local Refinement**: Search within 5km radius to find the node that minimizes actual road network distances
6. **Visualization**: Display the optimal location on a map

## Code Structure

- `main.py`: Entry point and pipeline orchestration
- `network.py`: Functions for loading and processing road networks
- `optimization.py`: Core optimization algorithms (Haversine and A*)
- `visualization.py`: Map plotting and result visualization
- `utils.py`: Helper functions and data processing utilities

## Example Results

![Optimization Result](./assets/result_visualization.png)

*Visualization of customer delivery locations (blue circles, scaled by frequency) and the optimal warehouse location (red star).*

## Performance

The GPU acceleration provides significant speedup compared to CPU-only implementations:

| Dataset Size | CPU Time | GPU Time | Speedup |
|--------------|----------|----------|---------|
| 1,000 points | 120s     | 8s       | 15x     |
| 10,000 points| 1800s    | 45s      | 40x     |
| 100,000 points| >8h     | 360s     | >80x    |

## Citing This Work

If you use this code in your research, please cite:

```
Lohar, R., Priyadarshi, P., & Reddy, N. Y. (2025). Optimal Location for an E-Commerce Warehouse. 
GitHub Repository. https://github.com/raj6515/Optimal-Location-for-an-E-Commerce-Warehouse
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [RAPIDS](https://rapids.ai/) for GPU-accelerated data science libraries
- [OSMnx](https://github.com/gboeing/osmnx) for easy OpenStreetMap network analysis
- [Boeing, G. (2017)](https://geoffboeing.com/publications/osmnx-complex-street-networks/) for OSMnx methods

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
