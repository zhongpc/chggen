"""Auxiliary functions for charge balance computation and manipulation."""
from __future__ import annotations
from typing import Tuple
import numpy as np
from pymatgen.core import Structure, Element
import torch

import cvxpy as cp



def charge_balance_from_structure(
    structure: Structure,
    oxi_state: dict,
    verbose: bool = False,
) -> dict:
    """Balance charge of the structure using quadratic optimization.
    Optimization constrains: charge balance
    Objective function (minimization): sum of squares of the number 
        of changed atoms
        
    Args:
        structure (Structure): pymatgen Structure to be balanced
        oxi_state (dict): oxidation state of the elements in structure
    
    Returns:
        (Dict): number of atoms to be added w.r.t. elem to the structure
    """
    charge_sub_ground = 0
    structure.add_oxidation_state_by_element(oxidation_states = oxi_state)
    charge_sub_add = charge_sub_ground - structure.charge
    charge_sub_add = int(charge_sub_add)
    if verbose:
        print(f"{charge_sub_add} positive charge need to be added to the subcell to match charge balance")
    
    # Quadratic optimization
    elems = [elem for elem in oxi_state.keys()]
    coeff = [val for val in oxi_state.values()]

    # Variables
    variables = [cp.Variable(integer=True) for _ in elems]

    # Objective function
    objective = cp.Minimize(sum(cp.square(var) for var in variables))

    # Constraints
    constraints = [sum(coeff[i]*variables[i] for i in range(len(elems))) == charge_sub_add]

    # Define the problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem using ECOS_BB solver
    prob.solve(solver=cp.ECOS_BB)

    result = {elem: np.round(var.value) for elem, var in zip(elems, variables)}

    # Print the result
    if verbose:
        print(result)
    
    return result

def structure_add_atoms(
    structure: Structure,
    atoms_to_add: dict,
    cut_threshold: float = 5.0,
    verbose: bool = False,
) -> Structure:
    """Add atoms to the structure."""
    # Get the lattice parameters
    lattice = structure.lattice.matrix

    # Check if the sphere completely encloses the unit cell
    diagonals = [np.dot(np.array([i, j, k]), lattice) for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]]
    max_diagonal = max(np.linalg.norm(diagonal) for diagonal in diagonals)
    if max_diagonal <= 2*cut_threshold:
        raise RuntimeError("The sphere completely encloses the unit cell")

    # Calculate the center of the cell
    center = np.dot(np.array([0.5, 0.5, 0.5]), lattice)

    # Add or remove atoms from the structure
    for atom, count in atoms_to_add.items():
        
        if count > 0:  # Add atoms
            for _ in range(int(count)):
                # Generate a random position in the unit cell but outside a sphere of cut
                while True:
                    position = np.random.rand(3)
                    position = np.dot(position, lattice)
                    if np.linalg.norm(position - center) > cut_threshold:  
                        break
                structure.append(Element(atom), position, coords_are_cartesian=True)
                if verbose:
                    print(f"Atom {atom} added at position {position}", 
                          f"(dis: {np.linalg.norm(position - center):.2f})")
                    
        elif count < 0:  # Remove atoms
            for _ in range(int(-count)):
                indices = [i for i, site in enumerate(structure) 
                           if site.specie.symbol == atom 
                           and np.linalg.norm(site.coords - center) > cut_threshold]
                if not indices:
                    raise RuntimeError(f"No atoms of type {atom} found in the structure outside the cut sphere")
                index_to_remove = np.random.choice(indices)
                if verbose:
                    print(f"{atom} atom removed at position {structure[index_to_remove].coords}", 
                          f"(dis: {np.linalg.norm(structure[index_to_remove].coords - center):.2f})")
                del structure[index_to_remove]
    
    return structure

def structure_to_tensor(
    structure: Structure,
    cut_threshold: float = 5.0,
) -> Tuple[torch.Tensor]:
    """Convert pymatgen Structure to torch tensor including
    angles, lengths, frac_coords, atom_types, num_atoms, mask."""
    # Construct framework structure
    lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
    angles = torch.tensor([structure.lattice.angles], dtype=torch.float32)
    lengths = torch.tensor([structure.lattice.lengths], dtype=torch.float32)
    frac_coords = torch.tensor(structure.frac_coords, dtype=torch.float32)

    # Compute the distance from current atom to the center of the cell
    center = torch.tensor([0.5, 0.5, 0.5], dtype = torch.float32)
    cart_dist = torch.norm((frac_coords - center)@lattice, dim=1)
    
    atom_types = []
    atom_masks = []
    for i, site in enumerate(structure.sites):
        atom_types.append(site.specie.Z)
        if cart_dist[i] < cut_threshold:
            atom_masks.append(0)
        else:
            atom_masks.append(1)

    atom_types = torch.tensor(atom_types, dtype=torch.int32)
    atom_masks = torch.tensor(atom_masks, dtype=torch.bool)
    num_atoms = torch.tensor([len(atom_types)], dtype=torch.int32)

    return (lengths, angles, frac_coords, atom_types, num_atoms, atom_masks)