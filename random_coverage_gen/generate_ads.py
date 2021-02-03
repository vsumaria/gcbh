"""
This script can be used to generate random configuration of adsorbates on any given surface at varied coverage.
This code will require using:
(1) https://gitlab.com/jgreeley-group/graph-theory-surfaces -- by Siddharth Deshpande
(2) https://github.com/GengSS/gcbh - checkatoms.py code 
This helps generate random configurations to start basin hopping simulations at various coverges to study the 
adsorbate/substrate system at different T/P conditions
Steps:
1> Generate clean surface with all atoms apart from the surface atoms (on which you will adsorb molecules) fixed. 
2> Add a molecule on this surface radomly in any position
3> Change "ads" variable --> example here using CO molecule and "nCO" defines the number of CO molecules
4> Change Bond ranges according to the atoms in the system
"""

import numpy as np
from numpy.linalg import norm
from ase.data import covalent_radii as covalent
from ase.neighborlist import NeighborList
from itertools import combinations
from ase.constraints import constrained_indices
import networkx as nx
import networkx.algorithms.isomorphism as iso
from pygcga.checkatoms import CheckAtoms
from chem_env_sid import process_atoms, process_site, unique_chem_envs, draw_atomic_graphs
import glob 

bond_match = iso.categorical_edge_match('bond', '')

def natural_cutoffs(atoms, multiplier=1.1):
    """Generate a neighbor list cutoff for every atom"""
    return [covalent[atom.number] * multiplier for atom in atoms]

def normalize(vector):
    return vector / norm(vector) if norm(vector) != 0 else vector * 0

def relative_position(atoms, neighbor, offset):
   return atoms[neighbor].position + np.dot(offset, atoms.get_cell())

def draw_networkx_many(graphs):
    import matplotlib.pyplot as plt
    for index, graph in enumerate(graphs):
        plt.figure(index, figsize=(2.5, 2.5))
        plt.axis('off')
        plt.title("Graph Index: {}".format(index))
        nx.draw_networkx(graph)
    plt.show()

def generate_normals(atoms, surface_normal=0.5, normalize_final=True, adsorbate_atoms=[]):
    normals = np.zeros(shape=(len(atoms), 3), dtype=float)

    atoms = atoms.copy()

    del atoms[adsorbate_atoms]

    cutoffs = natural_cutoffs(atoms)

    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    cell = atoms.get_cell()

    for index, atom in enumerate(atoms):
        normal = np.array([0, 0, 0], dtype=float)
        for neighbor, offset in zip(*nl.get_neighbors(index)):
            direction = atom.position - relative_position(atoms, neighbor, offset)
            normal += direction
        if norm(normal) > surface_normal:
            normals[index,:] = normalize(normal) if normalize_final else normal

    surface_mask = [index for index in range(len(atoms)) if norm(normals[index]) > 1e-5]

    return normals, surface_mask

def generate_site_type(atoms, surface_mask, normals, coordination, unallowed_elements=[]):
    cutoffs = natural_cutoffs(atoms)

    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    surface_mask = [index for index in surface_mask if atoms[index].symbol not in unallowed_elements]

    possible = list(combinations(set(surface_mask), coordination))
    valid = []
    sites = []

    for cycle in possible:
       for start, end in combinations(cycle, 2):
           if end not in nl.get_neighbors(start)[0]:
               break
       else: # All were valid
            valid.append(list(cycle))

    for cycle in valid:
        tracked = np.array(atoms[cycle[0]].position, dtype=float)
        known = np.zeros(shape=(coordination, 3), dtype=float)
        known[0] = tracked
        for index, (start, end) in enumerate(zip(cycle[:-1], cycle[1:])):
            for neighbor, offset in zip(*nl.get_neighbors(start)):
                if neighbor == end:
                    tracked += relative_position(atoms, neighbor, offset) - atoms[start].position
                    known[index + 1] = tracked

        average = np.average(known, axis=0)

        normal = np.zeros(3)
        for index in cycle:
            neighbors = len(nl.get_neighbors(index)[0]) ** 2
            normal += normals[index] * neighbors
        normal = normalize(normal)
        if coord ==2:
            average[2] = average[2] - 0.5
        if coord == 3:
            average[2] = average[2] -0.7
            #print(average)
            #print(average[2])
        site_ads =Site(cycle=cycle, position=average, normal=normal)
        sites.append(site_ads)
        
    return sites

def generate_site_graphs(atoms, full_graph, nl, sites, adsorbate_atoms=[], radius=3):
    cutoffs = natural_cutoffs(atoms)

    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    site_envs = [None] * len(sites)
    for index, site in enumerate(sites):
        new_site = process_site(atoms, full_graph, nl, site.cycle, radius=radius)
        site_envs[index] = [new_site]
        site.graph = site_envs[index]

    unique_envs, unique_sites = unique_chem_envs(site_envs, sites)

    return unique_sites

class Site(object):
    def __init__(self, cycle, position, normal, graph=None):
        self.cycle = cycle
        self.position = position
        self.normal = normal
        self.graph = graph

    def __eq__(self, other):
        return nx.is_isomorphic(self.graph, other.graph, edge_match=bond_match)

    def __repr__(self):
        return "Cycle:{}, Position:[{}, {}, {}], Normal:[{}, {}, {}], Graph:{}".format(
                       self.cycle, 
                       self.position[0], self.position[1], self.position[2],
                       self.normal[0], self.normal[1], self.normal[2],
                       self.graph)

    def adsorb(self, atoms, adsorbate, adsorbate_atoms, height=2):
        ads_copy = adsorbate.copy()
        ads_copy.rotate([0, 0, 1], self.normal, center=[0,0,0])
        ads_copy.translate(self.position + (self.normal*height))
        atoms.extend(ads_copy)

        index_to_check = range(len(atoms)-len(ads_copy), len(atoms))

        dist = float("inf")

        if len(adsorbate_atoms) != 0:
            for index in index_to_check:
                dists = atoms.get_distances(index, adsorbate_atoms, mic=True)
                dist = min(dist, dists.min())

        return dist

if __name__ == "__main__":
    from ase.visualize import view
    from ase.io import read, write
    from ase import Atoms
    from sys import argv
    from glob import glob
    from pathlib import Path
    
    movie = []
    all_unique = []

    for i in glob('*/'):
        if i=='__pycache__/':
            continue
        dire = i
        atoms = read(i+"/"+"POSCAR")
################# These can be added in as arg parsers ##################
        ads = read("CO.POSCAR")
        surface_atoms = ["Pt"]
        radii_multiplier = 1.1
        skin_arg = 0.25
        no_adsorb = ['']
        min_ads_dist = 2.4

        nl = NeighborList(natural_cutoffs(atoms, radii_multiplier), self_interaction=False,  bothways=True, skin=skin_arg)
        nl.update(atoms)

        adsorbate_atoms = [index for index, atom in enumerate(atoms) if atom.symbol not in surface_atoms]

        normals, mask = generate_normals(atoms,  adsorbate_atoms=adsorbate_atoms, normalize_final=True)   
        ### make sure to manually set the normals for 2-D materials, all atoms should have a normal pointing up, as all atoms are surface atoms
        #normals, mask = np.ones((len(atoms), 3)) * (0, 0, 1), list(range(len(atoms)))

        constrained = constrained_indices(atoms)
        mask = [index for index in mask if index not in constrained]
        #for index in mask:
        #    atoms[index].tag = 1

        atoms.set_velocities(normals/10)

        all_sites = []

        full_graph, envs = process_atoms(atoms, nl=nl, adsorbate_atoms=adsorbate_atoms,radius=3) 
        ### here the default radii as well as grid are considered, these can also be added as args.

        for coord in [1,2,3]:
            found_sites = generate_site_type(atoms, mask, normals, coordination=coord, unallowed_elements=no_adsorb)

            for site in found_sites:
                all_sites.append(site)

            unique_sites = generate_site_graphs(atoms, full_graph, nl, found_sites, adsorbate_atoms=adsorbate_atoms)

            for index, sites in enumerate(unique_sites):
                new = atoms.copy()
                for site in sites[0:1]:
                    ### this check is to ensure, that sites really close are not populated
                    if site.adsorb(new, ads, adsorbate_atoms) < min_ads_dist:
                        break
                    else:
                        all_unique.append(site)

# read clean surface on which CO/molecules need to adsorb
name = glob.glob('*.traj')
slab = read(name[0])
n=(len(all_unique))
pos_all = np.zeros([n,3])
normal_all = np.zeros([n,3])

for i,site in enumerate(all_unique):
    pos = site.position
    normal = site.normal
    pos_all[i,:] = pos
    normal_all[i,:] = normal

# np.savetxt('pos.txt',pos_all)
# np.savetxt('normal.txt',normal_all)

s_ = np.random.choice(len(pos_all),10)
max_trial=100
nCO = 22

bond_range={('C','Pt'):[1.2,10],('Pt','Pt'):[1,10.0],('C','C'):[2.3,10],('C','O'):[0.6,10],('Pt','O'):[1.5,10],('O','O'):[2.3,10]}

for _ in range(nCO):
    for _ in range(max_trial):
        s = np.random.choice(len(pos_all),1)[0]
        position = pos_all[s,:]
        normal = normal_all[s,:]
        height = 1.8
        ads_copy = ads.copy()
        ads_copy.rotate([0, 0, 1], normal, center=[0,0,0])
        ads_copy.translate(position + (normal*height))
        t_copy = slab.copy()
        slab.extend(ads_copy)
        
        inspector=CheckAtoms(bond_range=bond_range)
        if inspector.is_good(slab, quickanswer=True):
            break
        else:
            slab = t_copy.copy()

slab.write('start.traj')
