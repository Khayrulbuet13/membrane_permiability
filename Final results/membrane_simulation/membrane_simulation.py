# espressomd imports
import espressomd
from espressomd.interactions import FeneBond
from espressomd.interactions import AngleHarmonic

# other imports
import os, shutil
from utils import *  
import numpy as np



simulation_name = 'membrane_simulation'
simulation_folder = 'results/' + simulation_name

# delete folder if exists
if os.path.exists(simulation_folder):
    shutil.rmtree(simulation_folder)

# check if folder exists create if not
if not os.path.exists(simulation_folder): 
    os.makedirs(simulation_folder)

# copy simulation script to simulation folder
copy_file = simulation_name + ".py"
shutil.copy2(copy_file, simulation_folder)


# define system parameters
BOX_L = 50
BOX_W = 50
BOX_H = 100
time_step = 0.01

system = espressomd.System(box_l=(BOX_L, BOX_W, BOX_H))
system.time_step = time_step
system.cell_system.skin = 1.0

boundaries = create_box(BOX_L, BOX_W, BOX_H, simulation_folder)

# read atoms from gro file
mem_df, atom_names = read_gro_file('membrane_fiboblast.gro')
atom_types = mem_df['Atom'].unique()
mem_center = mem_df[['X', 'Y', 'Z']].mean().values

part_count = 0
particle_dict = {}
for i, atom in enumerate(atom_types, start=1):
    print(f'adding atoms with {atom} = {i} type')
    filtered_df = mem_df[mem_df['Atom'] == atom]
    particles = []
    for index, row in filtered_df.iterrows():
        system.part.add(id = part_count,   pos=[row['X'], row['Y'], row['Z']+ BOX_H/2 - mem_center[2]], type= i) 
        particles.append(part_count)
        part_count+= 1
    particle_dict[atom] = particles


# add diffusion particles
def generate_random_particles(n, box):
    particles = np.random.uniform(np.array([2,2,2]), box, size=(n, len(box)))
    return particles

rand_part = generate_random_particles(500, np.array([BOX_L-2, BOX_W-2, BOX_H/2 - mem_center[2]-3]))
rand_type = np.max(np.array([p.type for p in system.part])) + 1
for i in range(rand_part.shape[0]):
    system.part.add(id = part_count,   pos=[rand_part[i,0], rand_part[i,1], rand_part[i,2]], type= rand_type)
    part_count+= 1
        
system.non_bonded_inter[rand_type, rand_type].lennard_jones.set_params(epsilon=1.0, sigma=0.1, cutoff=0.95*2**(1./ 6), shift =0.25)       
atom_types = np.append(atom_types, 'R1')

from espressomd import lbboundaries
for boundary in boundaries:
    system.lbboundaries.add(lbboundaries.LBBoundary(shape=boundary, velocity=[0.0,0.0,0.0]))
    system.constraints.add(shape=boundary, particle_type=0, penetrable=False)

# system.non_bonded_inter[0, rand_type].lennard_jones.set_params(epsilon=1.0, sigma=0.1, cutoff=0.95*2**(1./ 6), shift =0.25)
system.non_bonded_inter[0, rand_type].lennard_jones.set_params(epsilon=1.0,
                                                       sigma=1.0,
                                                       cutoff=1.12246204831,
                                                       shift = 'auto')


#############################################################################################################
##################################### NON BONDED INTERACTIONS ###############################################
#############################################################################################################

particle_types=[1,2,3,4,5,6]
tail_types=[2,3,4,5]
head_types = [1, 6]
LJ_EPS = 1.0
LJ_SIG = 1
# Non-bonded LJ interactions
for i in particle_types:
    for j in particle_types:
        if i <= j:
            # Check the type of each particle
            #if i == 3 and j==4:
            #    continue
        
            if i in head_types and j in head_types:
                system.non_bonded_inter[i, j].lennard_jones.set_params(epsilon=LJ_EPS, sigma=LJ_SIG, cutoff=0.95*2**(1./ 6), shift =0.25)
                print(f"Head-Head interaction between {i} and {j}")
            elif (i in head_types and j in tail_types) or (i in tail_types and j in head_types):
                system.non_bonded_inter[i, j].lennard_jones.set_params(epsilon=LJ_EPS, sigma=LJ_SIG, cutoff=0.95*2**(1./ 6), shift =0.25)
                print(f"Head-Tail interaction between {i} and {j}")
            elif i in tail_types and j in tail_types:
                system.non_bonded_inter[i, j].lennard_jones.set_params(epsilon=LJ_EPS, sigma=LJ_SIG, cutoff=0.95*2**(1./ 6), shift =0.25)
                print(f"Tail-Tail interaction between {i} and {j}")

# Non-bonded LJ-COS2 interactions
for i_elem in particle_types:
    for j_elem in particle_types:
        #if i_elem == 3 and j_elem==4:
        #    continue
        if i_elem == j_elem:
            system.non_bonded_inter[i_elem, j_elem].lennard_jones_cos2.set_params(epsilon=1.0, sigma=LJ_SIG, width=1.6)
        if j_elem > i_elem:
            system.non_bonded_inter[i_elem, j_elem].lennard_jones_cos2.set_params(epsilon=1.0, sigma=LJ_SIG, width=1.6)
        print(f"tail-tail lj_cos2 interaction between {i_elem} and {j_elem}")

# with membrane gap on sigma = sigma + membrane_gap
#system.non_bonded_inter[3, 4].lennard_jones.set_params(epsilon=LJ_EPS, sigma=LJ_SIG + .2, cutoff=0.95*2**(1./ 6), shift =0.25)
#system.non_bonded_inter[3, 4].lennard_jones_cos2.set_params(epsilon=1.0, sigma=LJ_SIG + .2, width=1.6)
        
#############################################################################################################
######################################  BONDED INTERACTIONS #################################################
#############################################################################################################

# fene bond between head and tail
fene = FeneBond(k=30, d_r_max=1.5,r_0=0.0)
system.bonded_inter.add(fene)
for H1, T1, T2 in zip(particle_dict['H1'], particle_dict['T1'], particle_dict['T2']):
    system.part.by_id(H1).add_bond((fene, T1))
    system.part.by_id(T1).add_bond((fene, T2))
    
for H2, T3, T4 in zip(particle_dict['H2'], particle_dict['T3'], particle_dict['T4']):
    
    # because this membrane is flipped 
    system.part.by_id(T3).add_bond((fene, T4))
    system.part.by_id(T4).add_bond((fene, H2))
    

# angle_harmonic bond between head and tail
angle_harmonic = AngleHarmonic(bend=10.0, phi0= np.pi)
system.bonded_inter.add(angle_harmonic)
for H1, T1, T2 in zip(particle_dict['H1'], particle_dict['T1'], particle_dict['T2']):
    system.part.by_id(T1).add_bond((angle_harmonic,H1, T2))
for H2, T3, T4 in zip(particle_dict['H2'], particle_dict['T3'], particle_dict['T4']):
    system.part.by_id(T4).add_bond((angle_harmonic,H2, T3))   
    
    
#############################################################################################################
#
#                                   NVT Ensemble
#
#############################################################################################################


# Setting-up Thermosttat and Integrator
system.thermostat.turn_off()
system.integrator.set_vv() # same as set_nvt
system.thermostat.set_langevin(kT=0.1, gamma=2,seed=143)
steps = 50
name = 'nvt1'
sim_runner(system, steps, simulation_folder, name, atom_types)


#############################################################################################################
#
#                                   NVT Ensemble
#
#############################################################################################################


# Setting-up Thermosttat and Integrator
system.thermostat.turn_off()
system.integrator.set_vv() # same as set_nvt
system.thermostat.set_langevin(kT=0.2, gamma=2,seed=143)
steps = 50
name = 'nvt2'
sim_runner(system, steps, simulation_folder, name, atom_types)

#############################################################################################################
#
#                                   NVT Ensemble
#
#############################################################################################################


# Setting-up Thermosttat and Integrator
system.thermostat.turn_off()
system.integrator.set_vv() # same as set_nvt
system.thermostat.set_langevin(kT=0.4, gamma=2,seed=143)
steps = 5000
name = 'nvt3'
sim_runner(system, steps, simulation_folder, name, atom_types)

#############################################################################################################
#
#                                   NVT Ensemble
#
#############################################################################################################


# Setting-up Thermosttat and Integrator
system.thermostat.turn_off()
system.integrator.set_vv() # same as set_nvt
system.thermostat.set_langevin(kT=0.6, gamma=2,seed=143)
steps = 5000
name = 'nvt4'
sim_runner(system, steps, simulation_folder, name, atom_types)

#############################################################################################################
#
#                                   Production
#
#############################################################################################################
