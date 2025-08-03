import scipy.spatial as spatial
import pyvista as pv
import numpy as np
from scipy.interpolate import interp1d
import argparse
import sys


def reciprocal_lattice(real_lattice):
    """
    Compute reciprocal lattice vectors from real-space lattice vectors.

    Parameters
    ----------
    real_lattice : ndarray of shape (3, 3)
        real lattice vectors as a 3x3 matrix,
        where each column corresponds to a 
        lattice vector in cartesian coordinates

    Returns
    -------
    reciprocal : ndarray of shape (3, 3)
        reciprocal lattice vectors as a 3x3 matrix,
        where each column corresponds to a
        reciprocal lattice vector in cartesian coordinates
    """
    return 2 * np.pi * np.linalg.inv(real_lattice).T


def compute_brillouin_zone(reciprocal_lattice):
    """
    Compute the first Brillouin zone as the Voronoi cell of the reciprocal lattice.
    
    Parameters
    ----------
    reciprocal_lattice : ndarray of shape (3, 3)
        reciprocal lattice vectors as a 3x3 matrix
        
    Returns
    -------
    vor : scipy.spatial.Voronoi
        the Voronoi diagram corresponding to the first Brillouin zone, 
        computed from the positions in reciprocal space which are 
        a dot product of the reciprocal lattice vectors and the 3D
        grid of indices i, j, k in [-1, 0, 1]
    """
    indices = np.array([[i, j, k] for i in [-1, 0, 1] 
                                 for j in [-1, 0, 1] 
                                 for k in [-1, 0, 1]])
    points = np.array([np.dot(n, reciprocal_lattice) for n in indices])
    return spatial.Voronoi(points)

def parse_float_or_fraction(value):
    """
    Convert a string to float, handling fractional notation like '1/3'.
    
    Parameters
    ----------
    value : str
        String representing a number, can be a decimal or fraction
        
    Returns
    -------
    float
        The converted number
    """
    if '/' in value:
        numerator, denominator = value.split('/')
        return float(numerator) / float(denominator)
    else:
        return float(value)

# Dictionary of high symmetry points for different lattice types
HIGH_SYMMETRY_POINTS = {
    'FCC': {
        'Gamma': [0.0, 0.0, 0.0], 'X': [0.5, 0.0, 0.5], 'L': [0.5, 0.5, 0.5],
        'W': [0.5, 0.25, 0.75], 'K': [0.375, 0.375, 0.75], 'U': [0.625, 0.25, 0.625],
    },
    'BCC': {
        'Gamma': [0.0, 0.0, 0.0], 'H': [0.5, -0.5, 0.5], 
        'N': [0.0, 0.0, 0.5], 'P': [0.25, 0.25, 0.25],
    },
    'HCP': {
        'Gamma': [0.0, 0.0, 0.0], 'A': [0.0, 0.0, 0.5], 'M': [0.5, 0.0, 0.0],
        'K': [1/3, 1/3, 0.0], 'L': [0.5, 0.0, 0.5], 'H': [1/3, 1/3, 0.5],
    },
    'SC': {
        'Gamma': [0.0, 0.0, 0.0], 'X': [0.0, 0.5, 0.0],
        'M': [0.5, 0.5, 0.0], 'R': [0.5, 0.5, 0.5],
    },
    'TET': {
        'Gamma': [0.0, 0.0, 0.0], 'X': [0.0, 0.5, 0.0], 'M': [0.5, 0.5, 0.0],
        'Z': [0.0, 0.0, 0.5], 'R': [0.0, 0.5, 0.5], 'A': [0.5, 0.5, 0.5],
    },
    'BCT': {
        'Gamma': [0.0, 0.0, 0.0], 'N': [0.0, 0.5, 0.0], 'P': [0.25, 0.25, 0.25],
        'X': [0.0, 0.0, 0.5], 'M': [0.5, 0.5, -0.5], 'R': [0.5, 0.5, 0.5],
    },
    'RHL': {
        'Gamma': [0.0, 0.0, 0.0], 'L': [0.5, 0.0, 0.0], 'Z': [0.5, 0.5, 0.5],
        'F': [0.5, 0.5, 0.0], 'P': [0.25, 0.25, 0.25],
    },
    'ORC': {
        'Gamma': [0.0, 0.0, 0.0], 'X': [0.5, 0.0, 0.0], 'Y': [0.0, 0.5, 0.0],
        'Z': [0.0, 0.0, 0.5], 'S': [0.5, 0.5, 0.0], 'R': [0.5, 0.5, 0.5],
        'T': [0, 0.5, 0.5], 'U': [0.5, 0.0, 0.5],
    },
    'MCL': {
        'Gamma': [0.0, 0.0, 0.0], 'Y': [0.0, 0.5, 0.0], 'Z': [0.0, 0.0, 0.5],
        'A': [0.0, 0.5, 0.5], 'C': [0.5, 0.5, 0.0], 'D': [0.5, 0.0, 0.5],
        'E': [0.5, 0.5, 0.5],
    },
    'TRI': {
        'Gamma': [0.0, 0.0, 0.0], 'X': [0.5, 0.0, 0.0], 'Y': [0.0, 0.5, 0.0],
        'Z': [0.0, 0.0, 0.5], 'L': [0.5, 0.5, 0.0], 'M': [0.5, 0.0, 0.5],
        'N': [0.0, 0.5, 0.5], 'R': [0.5, 0.5, 0.5],
    }
}


def plot_brillouin_zone(vor, high_symmetry_cart, k_path=None, user_defined_points=None):
    """
    Plots the first Brillouin zone using the Voronoi diagram and
    highlights high-symmetry points and k-path.
    
    This function visualizes the Brillouin zone by rendering its faces, 
    marking high-symmetry points, and optionally drawing a user-defined 
    k-path through reciprocal space. If a k-path is provided, a smooth 
    path and an interactive marker along it are also generated.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        The Voronoi diagram representing the first Brillouin zone.

    high_symmetry_cart : dict of str of array-like
        Dictionary mapping high-symmetry point labels to their Cartesian coordinates.

    k_path : list of str, optional
        List of high symmetry point labels defining the k-path.
    """
    plotter = pv.Plotter(window_size=[800, 800])
    
    # Add Brillouin zone faces
    for region in vor.regions:
        if not -1 in region and region:  # Ignore infinite regions
            poly = [vor.vertices[i] for i in region]
            try:
                hull = spatial.ConvexHull(poly)
                
                # Extract faces from ConvexHull
                faces = []
                for simplex in hull.simplices:
                    faces.append([len(simplex)] + list(simplex))
                
                mesh = pv.PolyData(hull.points, faces)
                plotter.add_mesh(mesh, color='cyan', opacity=0.5)
            except spatial.QhullError:
                continue  # Skip degenerate faces
    
    # Add only the high-symmetry points that are used in the k-path or from input file
    points_to_show = set(k_path) if k_path is not None else set()
    # If we have user-defined points, add those too
    if user_defined_points is not None:
        points_to_show.update(user_defined_points.keys())
        
    for label, point in high_symmetry_cart.items():
        if label in points_to_show:
            plotter.add_point_labels(np.array([point]), [label],
                                  point_size=10, font_size=25,
                                  text_color='black')
    
    # Create k-path from the list of points
    if k_path is not None and len(k_path) > 1:
        # Create the path segments
        k_path_cart = []
        for point_label in k_path:
            if point_label in high_symmetry_cart:
                k_path_cart.append(high_symmetry_cart[point_label])
            else:
                print(f"Warning: Point {point_label} not found in high symmetry points.")
                return
        
        # Draw lines between points
        for i in range(len(k_path_cart) - 1):
            start = tuple(map(float, k_path_cart[i]))
            end = tuple(map(float, k_path_cart[i + 1]))
            line = pv.Line(start, end)
            plotter.add_mesh(line, color="black", line_width=3)
                
        # Create a dense path for smooth movement
        dense_path = []
        for i in range(len(k_path_cart) - 1):
            start = np.array(k_path_cart[i])
            end = np.array(k_path_cart[i + 1])
            segment_length = np.linalg.norm(end - start)
            
            n_points = max(int(segment_length * 50), 2)
            for t in np.linspace(0, 1, n_points): 
                point = start + t * (end - start)
                dense_path.append(point)
        
        # Convert to numpy array
        path_points = np.array(dense_path)
        
        # Now cumulative distances
        distances = [0.0]
        for i in range(1, len(path_points)):
            dist = np.linalg.norm(path_points[i] - path_points[i - 1])
            distances.append(distances[-1] + dist)
            
        total_length = distances[-1]
        
        # Interpolation function for smoother marker movement
        interp_func = interp1d(distances, path_points, axis=0,
                              kind='linear', fill_value="extrapolate")
            
        # Display the path with very small dots
        path_cloud = pv.PolyData(path_points)
        path_actor = plotter.add_mesh(path_cloud, color='blue', point_size=2, 
                                     render_points_as_spheres=True, opacity=0.3)
        
        # Create marker data structure that updates with the slider
        marker_data = pv.PolyData(path_points[0])
        marker_actor = plotter.add_mesh(marker_data, color='red', point_size=15,
                                      render_points_as_spheres=True)
        
        def update_position(value):
            target_distance = (value * total_length)
            position = interp_func(target_distance)     
            # Create new points - this is the key step
            marker_data.points[0] = position
            
            # Request an update
            marker_data.Modified()
            plotter.render()
        
        # Add slider widget
        slider = plotter.add_slider_widget(
            update_position,
            [0, 1],
            title="Position along k-path",
            value=0,
            pointa=(0.1, 0.1),
            pointb=(0.4, 0.1),
            style="modern"
        )
    
    plotter.show()

def read_lattice_and_sympoints(filename):
    """
    Read lattice type, vectors, and high symmetry points from a file.
    
    Parameters
    ----------
    filename : str
        Path to the input file
        
    Returns
    -------
    lattice_type : str
        The Bravais lattice type (e.g., 'FCC', 'BCC')
    lattice : ndarray of shape (3, 3)
        Real space lattice vectors
    high_sym_points : dict
        Dictionary of high symmetry points (optional)
    """
    try:
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            lattice_type = lines[0].upper()
            matrix_lines = lines[1:4]
            hs_lines = lines[4:]
            
            lattice = np.array([[float(x) for x in line.split()] for line in matrix_lines])
            if lattice.shape != (3, 3):
                raise ValueError("Expected a 3x3 matrix for lattice vectors.")
            
            # Process high-symmetry points: Initialize empty dictionary, process each line, and split into words
            high_sym_points = {}
            for line in hs_lines:
                parts = line.split()
                if len(parts) != 4:
                    continue  # Skip if line not label + 3 coordinates
                label, x, y, z = parts
                high_sym_points[label] = np.array([
                    parse_float_or_fraction(x), 
                    parse_float_or_fraction(y), 
                    parse_float_or_fraction(z)
                ])
                
            return lattice_type, lattice, high_sym_points
        
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    # Create parser with description, then add required parameters (with nargs='+' meaning one or more arguments)
    parser = argparse.ArgumentParser(description="Brillouin zone visualizer.")
    parser.add_argument("filename", help="Path to the .txt file with lattice and symmetry points")
    parser.add_argument("kpath", nargs='+', help="List of high symmetry points for k-path (e.g., Gamma X L)")
    
    return parser.parse_args()


def main():
    """
    Main function to run the Brillouin zone visualizer.

    Reads lattice vectors from a file and uses the specified k-path
    to visualize the Brillouin zone.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize empty list
    kpath_labels = []
    # Process each path argument
    for arg in args.kpath:
        # Remove any trailing commas
        arg = arg.rstrip(',')
        # Add to clean list
        kpath_labels.append(arg)
    
    # Read lattice information from file
    lattice_type, lattice, user_hs_points = read_lattice_and_sympoints(args.filename)
    
    # Determine which high symmetry points to use
    if not user_hs_points:
        if lattice_type not in HIGH_SYMMETRY_POINTS:
            print(f"Error: Lattice type '{lattice_type}' not recognized")
            return
        hs_points_frac = HIGH_SYMMETRY_POINTS[lattice_type]
    else:
        # Get standard high-symmetry points
        hs_points_frac = HIGH_SYMMETRY_POINTS.get(lattice_type, {}).copy()
        # Add user-defined points (overide standard points if same label)
        hs_points_frac.update(user_hs_points)
        
    
    # Compute reciprocal lattice and Brillouin zone
    recip_lattice = reciprocal_lattice(lattice)
    vor = compute_brillouin_zone(recip_lattice)
    
    # Convert fractional coordinates to Cartesian
    hs_points_cart = {label: np.dot(coords, recip_lattice) 
                     for label, coords in hs_points_frac.items()}
    
    # Plot the Brillouin zone with the specified k-path and user-defined points
    plot_brillouin_zone(vor, hs_points_cart, kpath_labels, user_hs_points)


if __name__ == '__main__':
    main()

    