"""Class for cutting subcell from a supercell denoised by CHGGEN."""

from __future__ import annotations
from typing import Any, Sequence
from tqdm import tqdm
import os
import random

import numpy as np

from pymatgen.core import Structure, Lattice, PeriodicSite
from pymatgen.analysis.ewald import EwaldSummation

class SubCellCut:
    """Class for cutting subcell from a supercell.
    
    Args:
        grid_size_vector (int): grid size for lattice vectors describing subcell.
        grid_size_origin (int): grid size for origins describing initial position.
        volume_range (Sequence): valid volume ranges for subcell.
    """
    
    def __init__(
        self,
        grid_size_vector: int,
        grid_size_origin: int,
        volume_range: Sequence,
        volume_deviation: float = 0.2,
        num_origin: int = 1,
       
    ) -> None:
        """Initialize a SubCellCut."""
        
        self.grid_size_vector = grid_size_vector
        self.grid_size_origin = grid_size_origin
        self.volume_range = volume_range
        self.volume_deviation = volume_deviation
        self.num_origin = num_origin
    
    def find_sub_structure(
        self,
        structure: Structure,
        cell_shape: str,
        pred_volume: float,
        max_num_structure: int = None,    
    ) -> list:
        """ Cut out subcells and find sub-structures given a large cell.

        Args:
            structure (Structure): structure of the large cell.
            cell_shape (str): shape of the subcell, i.e., cubic or skewed.
            pred_volume (float): average volume of the subcell.
            max_num_structure (int, optional): maximum number of 
                sub-structures to be found.
                Defaults = None.

        Returns:
            sub_structure_list (list): list of sub-structures fonud.
        """
        if cell_shape == 'cubic':
            vectors = self.generate_cubic_vectors_with_volume(
                grid_size = self.grid_size_vector,
                volume_range = self.volume_range,
            )
        
        formula_super = structure.composition.reduced_formula
        sub_structure_list =  []

        if max_num_structure is None:
            num_loop = len(vectors)
        else:
            num_loop = min(len(vectors), max_num_structure)
            
        # Loop over all possible subcells.
        for ii in tqdm(range(num_loop)):
            # Loop over all grid points for origins.

            # random select the origin point

            for jj in range(self.num_origin):
                ijk = random.choices(np.arange(0, grid_size), k = 3)
                
                # Select atoms with in subcell.
                origin = np.array(ijk)
                vector = vectors[ii]
                selected_atoms = [
                    site for site in structure.sites 
                    if self.is_point_in_parallelepiped(site.coords, origin, vector)
                ]
    
                # Generate sub-structure in unit cells.
                sublattice = Lattice(vector)
                new_atoms = [
                    PeriodicSite(site.specie, site.coords, sublattice, coords_are_cartesian = True) 
                    for site in selected_atoms
                ]
                try:
                    sub_structure = Structure.from_sites(new_atoms)
                except:
                    continue
                
                # Check if sub-structure is valid (volume and composition).
                formula_sub = sub_structure.composition.reduced_formula
                avg_volume = sub_structure.volume / sub_structure.num_sites
                volume_deviation = np.abs(avg_volume - pred_volume) / pred_volume
                
                if (formula_sub == formula_super) and (volume_deviation < 0.2):
                    sub_structure_list.append(sub_structure)

        return sub_structure_list
    
    @staticmethod
    def generate_skewed_vectors_with_volume(
        grid_size: int, 
        volume_range: Sequence,
    ) -> list[np.array]:
        """Generate skewed vectors with volume.

        Args:
            grid_size (int): grid size for lattice vectors describing subcell.
            volume_range (Sequence): range of valid volume for subcell. It should
                be a list or tuple of two elements, i.e., [min_volume, max_volume].
        
        Returns:
            subcells (list): list of skewed vectors formed subcells.
        """
        subcells = []
        # Loop over possible grid points to form cells.
        for a in range(1, grid_size + 1):
            for b in range(1, grid_size + 1):
                for c in range(1, grid_size + 1):
                    
                    # Considering off-diagonal elements for skewness
                    for d in range(0, grid_size // 2):
                        for e in range(0, grid_size // 2):
                            for f in range(0, grid_size // 2):

                                vec_a = np.array([a, d, e])
                                vec_b = np.array([d, b, f])
                                vec_c = np.array([e, f, c])

                                # Calculate the volume (determinant)
                                volume = np.linalg.det(np.array([vec_a, vec_b, vec_c]))
                                norm = np.array([np.linalg.norm(vec_a), np.linalg.norm(vec_b), np.linalg.norm(vec_c)])
                                if (volume > volume_range[0]) and \
                                   (volume < volume_range[1]) and \
                                   ((np.max(norm) - np.min(norm)) < np.mean(norm)/2 ):
                                    subcells.append(np.array([vec_a, vec_b, vec_c]))
        return subcells
    
    @staticmethod
    def generate_cubic_vectors_with_volume(
        grid_size: int,
        volume_range: Sequence,
    ) -> list[np.array]:
        """Generate cubic vectors with volume.

        Args:
            grid_size (int): size of the grid for lattice vectors describing subcell.
            volume_range (Sequence): range of valid volume for subcell. It should
                be a list or tuple of two elements, i.e., [min_volume, max_volume].

        Returns:
            supercell (list): list of cubic vectors formed subcells.
        """
        subcells = []
        # Loop over possible grid points to form cells. 
        for a in range(1, grid_size + 1):
            for b in range(1, grid_size + 1):
                for c in range(1, grid_size + 1):
                    
                    # Subcell vectors.
                    vec_a = np.array([a, 0, 0])
                    vec_b = np.array([0, b, 0])
                    vec_c = np.array([0, 0, c])

                    # Calculate the volume (determinant).
                    volume = np.linalg.det(np.array([vec_a, vec_b, vec_c]))
                    norm = np.array([np.linalg.norm(vec_a), np.linalg.norm(vec_b), np.linalg.norm(vec_c)])
                    if (volume > volume_range[0]) and \
                       (volume < volume_range[1]) and \
                       ((np.max(norm) - np.min(norm)) < np.mean(norm)/2 ):
                        subcells.append(np.array([vec_a, vec_b, vec_c]))
        return subcells
    
    @staticmethod
    def is_point_in_parallelepiped(
        point: np.array, 
        origin: np.array, 
        M: np.array,
    ) -> bool:
        """Check if a point is in a parallelepiped.

        Args:
            point (np.array): point to be checked.
            origin (np.array): origin of the parallelepiped.
            M (np.array): lattice matrix describing the parallelepiped.

        Returns:
            (bool): True if the point is in the parallelepiped, False otherwise.
        """
        # Compute fraction coordinates.
        M = M.T
        try:
            solution = np.linalg.solve(M, np.array(point) - np.array(origin))
        except np.linalg.LinAlgError:   # The system is unsolvable.
            return False
        
        # Check if the fraction coordinates is within the bounds [0, 1].
        return np.all(solution >= 0) and np.all(solution <= 1)
    
    def compute_ewald_energy(
        self,
        s_list: list[Structure],
    ) -> dict[dict]:
        """Compute the Ewald energy of a list of structures with guessed oxidation state."""
        se_list = []
        for structure in s_list:
            ewald_energy = self.compute_ewald_energy_single_structure(structure)
            se_dict = {'structure': structure, 'ewald_energy': ewald_energy/structure.num_sites}
            se_list.append(se_dict)
        se_list = sorted(se_list, key=lambda x: x['ewald_energy'])
        return se_list
    
    @staticmethod
    def compute_ewald_energy_single_structure(
        structure: Structure,
    ) -> float:
        """Compute the Ewald energy of a structure with guessed oxidation state."""
        structure.add_oxidation_state_by_guess()
        ewald_sum = EwaldSummation(structure)
        ewald_energy = ewald_sum.total_energy
        return ewald_energy
    
    def filter_ewald_energy(
        self,
        se_list: list[dict],
        energy_cutoff: float = -20,
    ) -> list[Structure]:
        """Filter structures with Ewald energy according to energy cutoff."""
        s_list_filtered = []
        for se_dict in se_list:
            if se_dict['ewald_energy'] < energy_cutoff:
                s_list_filtered.append(se_dict['structure'])
        return s_list_filtered
    
    def filter_min_distance(
        self,
        s_list: list[Structure],
        distance_cutoff: float = 1.5,
    ) -> list[Structure]:
        """Filter structures with minimum distance between atoms."""        
        s_list_filtered = []
        
        for structure in s_list:
            # Compute distance matrix.
            distance_matrix = structure.distance_matrix
            # Set the diagonal to infinity to ignore zero distances (distance of a site to itself)
            np.fill_diagonal(distance_matrix, np.inf)
            # Find the minimum distance
            min_distance = np.min(distance_matrix)
            
            if min_distance > distance_cutoff:
                s_list_filtered.append(structure)
        return s_list_filtered
    
    @staticmethod
    def save(
        s_list: list[Structure],
        folder_path: str,
    ) -> None:
        """Save structures to cif files in given folder."""
        mkdir(folder_path)
        for i, structure in enumerate(s_list):
            structure.to(filename = 'substructure_' + str(i) + '.cif')
        print(f"Totally {len(s_list)} structures generated and saved to {folder_path}.")
        
    def __call__(
        self,
        structure: Structure,
        cell_shape: str,
        pred_volume: float,
        max_num_structure: int = None, 
        energy_cutoff: float = -20,
        distance_cutoff: float = 1.5,  
        save_folder_path: str = './subcell_cut',
    ) -> None:
        """Cut out sub-structure in given large cell and filter with given criteria."""
        
        # Initial cut.
        s_list = self.find_sub_structure(
            structure = structure,
            cell_shape = cell_shape,
            pred_volume = pred_volume,
            max_num_structure = max_num_structure,
        )
        print(f"{len(s_list)} structures cut out.")
        
        # Compute ewald energy.
        se_list = self.compute_ewald_energy(s_list)
        
        # Filter ewald energy.
        s_list_filtered = self.filter_ewald_energy(se_list, energy_cutoff)
        print(f"{len(s_list_filtered)} structures left after ewald energy filtering.")
        
        # Filter minimum distance.
        s_list_filtered = self.filter_min_distance(s_list_filtered, distance_cutoff)
        print(f"{len(s_list_filtered)} structures left after minimum distance filtering.")
        
        # Save.
        # self.save(s_list_filtered, folder_path = save_folder_path)
        
        return s_list_filtered
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

def mkdir(
    folder_path: str,
) -> None:
    """Make directory if not exists."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
