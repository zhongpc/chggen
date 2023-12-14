
from pymatgen.core import Structure, Lattice, Species, Element, PeriodicSite

import numpy as np

from itertools import combinations, permutations
import random

s_super = Structure.from_file('./sc555_0.cif')

s_super.make_supercell([2,2,2])


formula_super = s_super.composition.reduced_formula


def generate_skewed_vectors_with_volume(grid_size):
    vectors = []
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
                            if (volume > 64) and (volume < 256) and ((np.max(norm) - np.min(norm)) < np.mean(norm)/2 ):
                                vectors.append(np.array([vec_a, vec_b, vec_c]))
    return vectors


def generate_cubic_vectors_with_volume(grid_size):
    vectors = []
    for a in range(1, grid_size + 1):
        for b in range(1, grid_size + 1):
            for c in range(1, grid_size + 1):
                # Considering off-diagonal elements for skewness


                d=0
                e=0
                f=0
                vec_a = np.array([a, d, e])
                vec_b = np.array([d, b, f])
                vec_c = np.array([e, f, c])

                # Calculate the volume (determinant)
                volume = np.linalg.det(np.array([vec_a, vec_b, vec_c]))
                norm = np.array([np.linalg.norm(vec_a), np.linalg.norm(vec_b), np.linalg.norm(vec_c)])
                if (volume > 64) and (volume < 256) and ((np.max(norm) - np.min(norm)) < np.mean(norm)/2 ):
                    vectors.append(np.array([vec_a, vec_b, vec_c]))
    return vectors




# Generate skewed vectors on a grid
grid_size = 12
skewed_vectors = generate_cubic_vectors_with_volume(grid_size)
# skewed_vectors = generate_skewed_vectors_with_volume(grid_size)

# Display the number of skewed vector sets found
len(skewed_vectors)


# Function to check if a point is within a parallelepiped defined by vectors
def is_point_in_parallelepiped(point, origin, M):
    # Create the matrix M from vectors A, B, C
    M = M.T

    # Solve for u, v, w
    try:
        solution = np.linalg.solve(M, np.array(point) - np.array(origin))
    except np.linalg.LinAlgError:
        # The system is unsolvable (e.g., vectors are linearly dependent)
        return False

    # Check if the solution is within the bounds [0, 1] for u, v, w
    return np.all(solution >= 0) and np.all(solution <= 1)




def find_sub_structure(s_super, skewed_vectors, grid_size = 12, pred_volume = 12.8, max_number_structure = 100):

    formula_super = s_super.composition.reduced_formula

    sub_structure_list =  []




    for ii in range(len(skewed_vectors)):

        grid_size = 6

        for i in range(0,grid_size):
            for j in range(0,grid_size):
                for k in range(0,grid_size):

                    origin = np.array([i,j,k])

                    skewvectors = skewed_vectors[ii]

                    # Define all cart_coords
                    cart_coords = s_super.cart_coords


                    selected_atoms = [site for site in s_super.sites if is_point_in_parallelepiped(site.coords, origin, skewvectors)]

                    sublattice = Lattice(skewvectors)

                    new_atoms = [PeriodicSite(site.specie, site.coords, sublattice, coords_are_cartesian = True) for site in selected_atoms]
                    try:
                        sub_structure = Structure.from_sites(new_atoms)
                    except:
                        print("No atom")

                        continue

                    formula_sub = sub_structure.composition.reduced_formula

                    avg_volume = sub_structure.volume / sub_structure.num_sites
                    volume_deviation = np.abs(avg_volume - pred_volume) / pred_volume

                    if (formula_sub == formula_super) and (volume_deviation < 0.2):
                        sub_structure_list.append(sub_structure)

                        print("Having {} structures.".format(len(sub_structure_list)) )

                        if len(sub_structure_list) > max_number_structure:
                            return sub_structure_list


    return sub_structure_list



sub_structure_list = find_sub_structure(s_super= s_super,
                                        skewed_vectors= skewed_vectors,
                                        grid_size= 12,
                                        pred_volume = 12.8)




from pymatgen.analysis.ewald import EwaldSummation


# In[12]:


structure = sub_structure_list[0]
s_list = []

for structure in sub_structure_list:


    structure.add_oxidation_state_by_guess()

    ewald_sum = EwaldSummation(structure)
    ewald_energy = ewald_sum.total_energy


    s_dict = {'structure': structure, 'ewald_energy': ewald_energy / structure.num_sites}

    s_list.append(s_dict)



get_ipython().system('rm -f test*')

sorted_s_list = sorted(s_list, key=lambda x: x['ewald_energy'])

for ii, s_dict in enumerate(sorted_s_list):
    s_save = s_dict['structure']
    ewald_energy = s_dict['ewald_energy']
    structure = s_save # Example for loading from a CIF file

    # Compute the distance matrix
    distance_matrix = structure.distance_matrix

    # Set the diagonal to infinity to ignore zero distances (distance of a site to itself)
    np.fill_diagonal(distance_matrix, np.inf)

    # Find the minimum distance
    min_distance = np.min(distance_matrix)
#     print(min_distance, ewald_energy)

    if (min_distance > 1.5) and (ewald_energy < -20):
        print(ewald_energy)

        s_save.to(filename='test' + str(np.round(ewald_energy, 3)) + '.cif')


# In[ ]:





# In[17]:


s_fine = Structure.from_file('./fine_diff0.cif')
s_fine.add_oxidation_state_by_guess()

ewald_sum = EwaldSummation(s_fine)
ewald_energy = ewald_sum.total_energy / s_fine.num_sites
