
import scipy.spatial as spatial
import pyvista as pv
import numpy as np
from scipy.interpolate import interp1d, griddata
from scipy.spatial.distance import cdist
import argparse
import sys
import os



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


def read_band_data(filename):
    '''
    Reads band structure data from a text file, including:
    - k-points (fractional coordinates)
    - eigenvalues for each band at each k-point
    - the Fermi energy
    - the reciprocal lattice vectors

    The input file is expected to contain lines in the following format:
    - A line beginning with '# Fermi energy:' followed by the Fermi energy value.
    - Three lines starting with '# Reciprocal lattice vectors' followed by 3D vectors (one per line).
    - Data lines with: 3 fractional k-point coordinates followed by eigenvalues for each band.
    - Comment lines starting with '#' are ignored (unless they mark the Fermi energy or reciprocal lattice section).

    Parameters
    ----------
    filename : str
        Path to the band structure data file.

    Returns
    -------
    kpoints_cart : np.ndarray
        A (num_kpoints, 3) array of k-points in Cartesian coordinates.

    eigenvalues : np.ndarray
        A (num_kpoints, num_bands) array of eigenvalues at each k-point.

    fermi_energy : float
        The Fermi energy read from the file.

    reciplattice : np.ndarray
        A (3, 3) array representing the reciprocal lattice vectors (each row is a vector).

    Raises
    ------
    ValueError
        If the reciprocal lattice vectors are not properly read or do not form a 3×3 array.

    Warnings
    --------
    - Warns if data lines do not contain enough entries.
    - Warns if no data is read for k-points or eigenvalues.
    - Prints errors encountered while parsing malformed lines.

    '''
    kpoints_frac = []
    eigenvalues = []
    fermi_energy = None
    reciplattice = []

    with open(filename, 'r') as f:
        reading_recip_lattice = False
        recip_lines_read = 0

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('# Fermi energy'):
                try:
                    fermi_energy = float(line.split(':')[1].strip())
                except Exception as e:
                    print(f"Error reading Fermi energy: {e}")
                continue

            if line.startswith('# Recirpcal lattice vectors') or line.startswith('# Reciprocal lattice vectors'):
                reading_recip_lattice = True
                reciplattice = []
                recip_lines_read = 0
                continue

            if reading_recip_lattice:
                try:
                    vec = list(map(float, line.lstrip('#').strip().split()))
                    if len(vec) != 3:
                        print(f"Warning: Reciprocal lattice vector line does not have 3 components: {line}")
                    reciplattice.append(vec)
                    recip_lines_read += 1
                    if recip_lines_read == 3:
                        reciplattice = np.array(reciplattice)
                        reading_recip_lattice = False
                except Exception as e:
                    print(f"Error reading reciprocal lattice vector line: {line} -> {e}")
                continue

            if line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 4:
                print(f"Warning: Data line does not have at least 4 values (3 k-points + eigenvalues): {line}")
                continue

            try:
                k = list(map(float, parts[:3]))
                eigs = list(map(float, parts[3:]))
            except Exception as e:
                print(f"Error parsing line: {line} -> {e}")
                continue

            kpoints_frac.append(k)
            eigenvalues.append(eigs)

    if len(kpoints_frac) == 0 or len(eigenvalues) == 0:
        print("Warning: No k-points or eigenvalues data read.")

    kpoints_frac = np.array(kpoints_frac)
    eigenvalues = np.array(eigenvalues)

    if reciplattice is None or len(reciplattice) != 3 or reciplattice.shape != (3, 3):
        raise ValueError(f"Reciprocal lattice vectors not properly read: shape {reciplattice.shape if reciplattice is not None else 'None'}")

    # Convert fractional k-points to Cartesian coordinates
    kpoints_cart = np.dot(kpoints_frac, reciplattice)

    return kpoints_cart, eigenvalues, fermi_energy, reciplattice


def analyse_band_structure(eigenvalues, fermi_energy, kpoints_cart):
    '''
    Analyse a band structure to find key electronic properties.

    This function determines:
      - The valence band maximum (VBM)
      - The conduction band minimum (CBM)
      - The band gap value
      - Whether the band gap is direct or indirect
      - The k-point locations and band indices where VBM and CBM occur

    Parameters
    ----------
    eigenvalues : np.ndarray
        A 2D array of shape (num_kpoints, num_bands) containing the energy eigenvalues
        at each k-point for all bands.

    fermi_energy : float
        The Fermi energy. Used to separate occupied and unoccupied states.

    kpoints_cart : np.ndarray
        A 2D array of shape (num_kpoints, 3) with the Cartesian coordinates of each k-point.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'vbm': float, energy of the valence band maximum
        - 'cbm': float, energy of the conduction band minimum
        - 'band_gap': float, value of the band gap (cbm - vbm)
        - 'vbm_kpoints': np.ndarray, Cartesian coordinates of k-points where VBM occurs
        - 'cbm_kpoints': np.ndarray, Cartesian coordinates of k-points where CBM occurs
        - 'vbm_band_idx': np.ndarray, indices of bands containing the VBM
        - 'cbm_band_idx': np.ndarray, indices of bands containing the CBM
        - 'min_dist': float, minimum distance between VBM and CBM k-points
        - 'is_direct': bool, True if the band gap is direct (VBM and CBM occur at the same k-point)

    Warnings
    --------
    - If no occupied or unoccupied states are found relative to the Fermi energy, a warning is printed and None is returned.
    
    '''
    
    # Find VBM and CBM, compute band gap
    occupied_states = eigenvalues[eigenvalues <= fermi_energy]
    unoccupied_states = eigenvalues[eigenvalues > fermi_energy]
    
    if len(occupied_states) == 0:
        print("Warning: No occupied states found!")
        return None
    if len(unoccupied_states) == 0:
        print("Warning: No unoccupied states found!")
        return None
        
    vbm = occupied_states.max()
    cbm = unoccupied_states.min()
    band_gap = cbm - vbm  
    
    # Makes Boolean masks where element is true if eigenvalues numerically close to vbm or cbm
    vbm_mask = np.isclose(eigenvalues, vbm, atol=1e-6)
    cbm_mask = np.isclose(eigenvalues, cbm, atol=1e-6)
    
    # Find k-point indices where VBM and CBM occur, np.where() returns a tuple so [0] grabs array
    vbm_kpoint_indices = np.where(np.any(vbm_mask, axis=1))[0]  # Any band at each k-point (checks each row)
    cbm_kpoint_indices = np.where(np.any(cbm_mask, axis=1))[0]  # Any band at each k-point(checks each row)
    
    # Extract k-point positions for VBM and CBM in Cartesian coordinates
    vbm_kpoints = kpoints_cart[vbm_kpoint_indices]
    cbm_kpoints = kpoints_cart[cbm_kpoint_indices]
    
    # Find the bands that contain VBM and CBM, np.where returns a tuple so [0] grabs array
    vbm_band_idx = np.where(np.any(vbm_mask, axis=0))[0]  # Any k-point in each band (checks each column)
    cbm_band_idx = np.where(np.any(cbm_mask, axis=0))[0]  # Any k-point in each band (checks each column)
    
    # Determine if band gap is direct or indirect
    # Check if arrays of Cartesian coordinates for k-points > 0 
    if len(vbm_kpoints) > 0 and len(cbm_kpoints) > 0:
        # Compute the Euclidean distance between each VBM k-point and each CBM k-point
        distances = cdist(vbm_kpoints, cbm_kpoints)
        # Find minimum distance between any VBM and CBM k-point
        min_dist = np.min(distances)
        # Check if minimum distance is effectively zero, handle floating-point rounding issues by allowing small tolerance
        is_direct = np.isclose(min_dist, 0.0, atol=1e-6)
    # If no valid VBM or CBM k-points, assigns min_dist to infinity and sets is_direct to False
    else:
        min_dist = float('inf')
        is_direct = False
    
    return {
        'vbm': vbm,
        'cbm': cbm,
        'band_gap': band_gap,
        'vbm_kpoints': vbm_kpoints,
        'cbm_kpoints': cbm_kpoints,
        'vbm_band_idx': vbm_band_idx,
        'cbm_band_idx': cbm_band_idx,
        'min_dist': min_dist,
        'is_direct': is_direct
    }

def create_isoenergetic_surface(kpoints_cart, eigenvalues, band_data, recip_lattice,
                                cube_size=0.02, vbm_tol=0.01, cbm_tol=0.01):
    """
    Create isoenergetic surfaces by placing mini cubes around k-points 
    that have eigenvalues within tolerance of VBM and CBM.
    
    Parameters
    ----------
    kpoints_cart : ndarray
        K-points in Cartesian coordinates
    eigenvalues : ndarray
        Eigenvalues at each k-point
    band_data : dict
        Band structure analysis results
    recip_lattice : ndarray
        Reciprocal lattice vectors
    cube_size : float, default=0.02
        Size of the mini cubes in reciprocal space units
    vbm_tol : float, default=0.1
        Energy tolerance for VBM isosurface (eV)
    cbm_tol : float, default=0.1
        Energy tolerance for CBM isosurface (eV)
        
    Returns
    -------
    dict
        Dictionary containing VBM and CBM isosurfaces as PyVista meshes.
    """
    
    # Get VBM and CBM values
    vbm_energy = band_data['vbm']
    cbm_energy = band_data['cbm']
    
    # Makes Boolean masks where element is true if eigenvalue within tolerance of the VBM or CBM energy
    vbm_mask = np.abs(eigenvalues - vbm_energy) <= vbm_tol 
    cbm_mask = np.abs(eigenvalues - cbm_energy) <= cbm_tol 
    
    # Find k-point indices where VBM and CBM occur, np.where() returns a tuple so [0] grabs array
    vbm_kpoint_indices = np.where(np.any(vbm_mask, axis=1))[0] # Any band at each k-point (checks each row)
    cbm_kpoint_indices = np.where(np.any(cbm_mask, axis=1))[0] # Any band at each k-point (checks each row)
    
    # Create VBM isosurface from cubes
    vbm_cubes = []
    if len(vbm_kpoint_indices) > 0:
        vbm_kpoints = kpoints_cart[vbm_kpoint_indices]
        for kpoint in vbm_kpoints:
            cube = pv.Cube(center=kpoint, x_length=cube_size, 
                          y_length=cube_size, z_length=cube_size)
            vbm_cubes.append(cube)
        
        # Merge all VBM cubes into one mesh
        if len(vbm_cubes) > 1:
            # Start with first cube as the base mesh to merge into
            vbm_isosurface = vbm_cubes[0]
            # Iterate over the rest of the cubes (skipping first one), for each cube merge it into the current vbm_isosurface
            for cube in vbm_cubes[1:]:
                vbm_isosurface = vbm_isosurface.merge(cube)
        else:
            # If theres only one cube, use directly. No merging necessary
            vbm_isosurface = vbm_cubes[0]
    else:
        # Create empty mesh if no points found
        vbm_isosurface = pv.PolyData()
    
    # Create CBM isosurface from cubes
    cbm_cubes = []
    if len(cbm_kpoint_indices) > 0:
        cbm_kpoints = kpoints_cart[cbm_kpoint_indices]
        for kpoint in cbm_kpoints:
            cube = pv.Cube(center=kpoint, x_length=cube_size, 
                          y_length=cube_size, z_length=cube_size)
            cbm_cubes.append(cube)
        
        # Merge all CBM cubes into one mesh
        if len(cbm_cubes) > 1:
            # Start with first cube as the base mesh to merge into
            cbm_isosurface = cbm_cubes[0]
            # Iterate over the rest of the cubes (skipping first one), for each cube merge it into the current cbm_isosurface
            for cube in cbm_cubes[1:]:
                cbm_isosurface = cbm_isosurface.merge(cube)
        else:
            # If theres only one cube, use directly. No merging necessary
            cbm_isosurface = cbm_cubes[0]
    else:
        # Create empty mesh if no points found
        cbm_isosurface = pv.PolyData()
    
    return {
        'vbm_isosurface': vbm_isosurface,
        'cbm_isosurface': cbm_isosurface,
        'vbm_kpoint_count': len(vbm_kpoint_indices),
        'cbm_kpoint_count': len(cbm_kpoint_indices)
    }


def plot_brillouin_zone(vor, high_symmetry_cart, k_path=None, user_defined_points=None, 
                        band_data=None, kpoints_cart=None, isosurfaces=None):
    """
    Plots the first Brillouin zone using the Voronoi diagram and
    highlights high-symmetry points and k-path.
    
    This function visualises the Brillouin zone by rendering its faces, 
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
    try:
        plotter = pv.Plotter(window_size=[800, 800])
        print("PyVista plotter created successfully")
    except Exception as e:
        print(f"Error creating PyVista plotter: {e}")
        return
    
    # Add Brillouin zone faces
    faces_added = 0
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
                faces_added += 1
            except spatial.QhullError:
                continue  # Skip degenerate faces
    
    # Add isoenergetic surfaces if available
    if isosurfaces is not None:
        if isosurfaces['vbm_isosurface'].n_points > 0:
            plotter.add_mesh(isosurfaces['vbm_isosurface'], color='red', opacity=0.8,
                             label='VBM_Isosurface')
            print(f"VBM isosurface created with {isosurfaces['vbm_kpoint_count']} k-points")
        else:
            print("No VBM isosurface found in given range")
            
        if isosurfaces['cbm_isosurface'].n_points > 0:
            plotter.add_mesh(isosurfaces['cbm_isosurface'], color='blue', opacity=0.8,
                             label='CBM_Isosurface')
            print(f"CBM isosurface created with {isosurfaces['cbm_kpoint_count']} k-points")
        else:
            print("No CBM isosurface found in given range")
          
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
        
    if band_data is not None:

        gap_type = "Direct" if band_data['is_direct'] else "Indirect"
        info_text = (f"Band Gap: {band_data['band_gap']:.3f} eV ({gap_type})\n"
                     f"VBM: {band_data['vbm']:.3f} eV (Red)\n"
                     f"CBM: {band_data['cbm']:.3f} eV (Blue)\n"
                     f"Minimum k-space distance between VBM and CBM: {band_data['min_dist']:.6f} 1/Å")
         
        plotter.add_text(info_text, position='upper_right', font_size=20, color='black')        
    
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
        path_actor = plotter.add_mesh(path_cloud, color='black', point_size=2, 
                                     render_points_as_spheres=True, opacity=0.3)
        
        # Create marker data structure that updates with the slider
        marker_data = pv.PolyData(path_points[0])
        marker_actor = plotter.add_mesh(marker_data, color='orange', point_size=15,
                                      render_points_as_spheres=True, opacity=1.0)
        
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
    
    #plotter.add_axes()
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
            
            # Process high-symmetry points: Initialise empty dictionary, process each line, and split into words
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
    parser = argparse.ArgumentParser(description="Brillouin zone visualiser.")
    parser.add_argument("filename", help="Path to the .txt file with lattice and symmetry points")
    parser.add_argument("kpath", nargs='+', help="List of high symmetry points for k-path (e.g., Gamma X L)")
    parser.add_argument("--bands", "-b", help="Path to band structure file (optional)", default=None)
    parser.add_argument("--isosurfaces", "-i", action="store_true",
                        help="Create isoenergetic surfaces for VBM and CBM (requires --bands)")
    
    return parser.parse_args()


def main():
    """
    Main function to run the Brillouin zone visualiser.

    Reads lattice vectors from a file and uses the specified k-path
    to visualise the Brillouin zone.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialise variables
    kpath_labels = []
    band_data = None
    kpoints_cart = None
    isosurfaces = None
    
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
            print(f"Error: Lattice type '{lattice_type}' not recognised")
            return
        hs_points_frac = HIGH_SYMMETRY_POINTS[lattice_type]
    else:
        # Get standard high-symmetry points
        hs_points_frac = HIGH_SYMMETRY_POINTS.get(lattice_type, {}).copy()
        # Add user-defined points (override standard points if same label)
        hs_points_frac.update(user_hs_points)
        
    # Handle band structure data
    if args.bands and os.path.exists(args.bands):
        try:
            kpoints_cart, eigenvalues, fermi_energy, recip_lattice_from_file = read_band_data(args.bands)
            
            # Use reciprocal lattice from band file
            recip_lattice = recip_lattice_from_file
            
            band_data = analyse_band_structure(eigenvalues, fermi_energy, kpoints_cart)
            
            print(f"Band structure analysis:")
            print(f"VBM: {band_data['vbm']:.3f} eV")
            print(f"CBM: {band_data['cbm']:.3f} eV")
            print(f"Band gap: {band_data['band_gap']:.3f} eV")
            gap_type = "Direct" if band_data['is_direct'] else "Indirect"
            print(f"Gap type: {gap_type}")
            print(f"Minimum k-space distance between VBM and CBM: {band_data['min_dist']:.6f} 1/Å")            
            
            # Create isosurfaces if requested
            if args.isosurfaces:
                isosurfaces = create_isoenergetic_surface(kpoints_cart, eigenvalues, band_data, recip_lattice)
                if isosurfaces is None:
                    print("Warning: Failed to create isoenergetic surfaces")
                else:
                    print(f"Isoenergetic surfaces created successfully")
            
        except Exception as e:
            print(f"Error reading band structure file: {e}")
            print("Continuing without band structure data...")
            recip_lattice = reciprocal_lattice(lattice)
    elif args.bands:
        print(f"Warning: Band structure file '{args.bands}' not found")
        recip_lattice = reciprocal_lattice(lattice)
    else:
        recip_lattice = reciprocal_lattice(lattice)

    # Compute Brillouin zone
    vor = compute_brillouin_zone(recip_lattice)

    # Convert fractional coordinates to Cartesian
    hs_points_cart = {label: np.dot(coords, recip_lattice) 
                 for label, coords in hs_points_frac.items()}
    
    
    # Plot the Brillouin zone with the specified k-path and user-defined points
    plot_brillouin_zone(vor, hs_points_cart, kpath_labels, user_hs_points, band_data, kpoints_cart, isosurfaces)
    
    
if __name__ == '__main__':

    main()
