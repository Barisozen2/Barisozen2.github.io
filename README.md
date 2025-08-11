# Barisozen2.github.io
# Brillouin Zone Isosurface Visualiser BZISOsurf

A Python tool for visualising the first Brillouin zone, high-symmetry points, high-symmetry point paths and isoenergetic surfaces of band extrema in 3D.

https://barisozen2.github.io/hcp_brillouin_zone.html

*example visualisation of HCP Brillouin zone (without interactive elements)* 

https://barisozen2.github.io/Si_fcc_BZ_isosurf(GLLB-SC).html

*example visualisation of FCC Brillouin zone and Silicon isoenergetic surface of band extrema (without interactive elements)*

https://barisozen2.github.io/MoS2_hcp_BZ_isosurf(GLLB-SC).html

*example visualisation of HCP Brillouin zone and Molybdenite isoenergetic surface of band extrema (without interactive elements)*

## Features
- **3D Brillouin Zone Construction** from reciprocal lattice vectors
- **High-symmetry point paths** for common lattices (FCC, BCC, HCP, etc.)
- **Band structure analysis**:
  - Valence Band Maximum (VBM) / Conduction Band Minimum (CBM) detection
  - Direct/indirect band gap classification
  - Isoenergetic surface generation
- **Interactive visualisation** with k-path exploration slider

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Barisozen2/Barisozen2.github.io:

3. Navigate into the repository:
   ```bash
   cd Barisozen2.github.io 

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
1. Basic BZ and k-path visualisation  
   ```bash
   python BZISOvis.py lattice_input.txt Gamma X L

- lattice_input.txt: File containing lattice vectors (see Input Format)
- Gamma X L: High-symmetry points defining the k-path

2. With Band Structure Data
   ```bash
   python BZISOvis.py lattice_input.txt Gamma X L --bands band_data.txt  
    
3. Generate Isoenergetic Surfaces
   ```bash
   python BZISOvis.py lattice_input.txt Gamma X L --bands band_data.txt --isosurfaces

## Input Formats
1. Lattice Input file
   ```text
   FCC
   0.5 0.5 0
   0.5 0 0.5
   0 0.5 0.5
   Gamma 0 0 0
   X 0.5 0 0.5
   W 0.5 0.25 0.75 
   K 0.375 0.375 0.75
   
- First line: Lattice type (e.g., FCC, HCP, SC, BCC)
- Next 3 lines: Real-space lattice vectors (3x3 matrix)
- Remaining lines (optional): Custom high-symmetry points (Label x y z)

2. Band Data File
   ```text
   # Fermi energy: 5.0
   # Reciprocal lattice vectors
   1.0 0.0 0.0
   0.0 1.0 0.0
   0.0 0.0 1.0
   0.0 0.0 0.0 -2.3 -1.8 ...
   0.1 0.0 0.0 -2.1 -1.6 ...

- Fermi energy and reciprocal lattice vectors marked with #
- Data lines: k_x k_y k_z eigenvalue1 eigenvalue2 ...

## Output
1. Interactive 3D plot showing:
  - Brillouin zone (transparent cyan)
  - High-symmetry points (labeled)
  - VBM (red) and CBM (blue) isosurfaces (if enabled)
2. Terminal output with band structure analysis:
   ```text
   Band gap: 1.23 eV (Indirect)
   VBM: -0.5 eV at [0.0, 0.0, 0.0]
   CBM: 0.73 eV at [0.5, 0.5, 0.5] 

## Examples
1. FCC Lattice
   ```bash 
   python BZISOvis.py examples/fcc.txt Gamma X W K L U

2. Silicon Isoenergetic Surface
   ```bash
   python BZISOvis examples/fcc.txt Gamma X --bands examples/Si_GLLBSC_band_data.txt --isosurfaces

3. Molybdenite Isoenergetic surface
   ```bash
   python BZISOvis examples/fcc.txt Gamma X --bands examples/MoS2_GLLBSC_band_data.txt --isosurfaces

## Dependencies
- Python 3.7+
- numpy, scipy, pyvista, argparse

## Licsense
- MIT Licsence. See [LICENSE](LICENSE) for details
