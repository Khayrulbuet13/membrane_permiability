import pandas as pd
import numpy as np
import os
import time


from espressomd import shapes
import object_in_fluid as oif
def create_box(BOX_L,BOX_W,BOX_H, directory):
        length  = BOX_L
        width  = BOX_W
        hight  = BOX_H

        # check if folder exists create if not
        directory = directory + '/vtk'
        if not os.path.exists(directory): 
            os.makedirs(directory)
        
        boundaries=[]
        # left wall
        tmp_shape = shapes.Rhomboid(corner=[0.0, 0.0, 0.0],a=[1.0, 0.0, 0.0],b=[0.0, width, 0.0],c=[0.0, 0.0, hight], direction=1)
        boundaries.append(tmp_shape)
        oif.output_vtk_rhomboid(rhom_shape=tmp_shape, out_file=directory + '/wallLeft.vtk')

        # Rigth wall
        tmp_shape = shapes.Rhomboid(corner=[length-1, 0.0, 0.0],a=[1, 0.0, 0.0],b=[0.0, width, 0.0],c=[0.0, 0.0, hight], direction=1)
        boundaries.append(tmp_shape)
        oif.output_vtk_rhomboid(rhom_shape=tmp_shape, out_file=directory + '/wallRigth.vtk')
        
        
        
        # bottom wall
        tmp_shape = shapes.Rhomboid(corner=[0.0, 0.0, 0.0],a=[length, 0.0, 0.0],b=[0.0, width, 0.0],c=[0.0, 0.0, 1.0], direction=1)
        boundaries.append(tmp_shape)
        oif.output_vtk_rhomboid(rhom_shape=tmp_shape, out_file=directory + '/wallBottom.vtk')

        # top wall
        tmp_shape = shapes.Rhomboid(corner=[0.0, 0.0, hight],a=[length, 0.0, 0.0],b=[0.0, width, 0.0],c=[0.0, 0.0, - 1.0], direction=1)
        boundaries.append(tmp_shape)
        oif.output_vtk_rhomboid(rhom_shape=tmp_shape, out_file=directory + '/wallTop.vtk')

        # Front wall
        tmp_shape = shapes.Rhomboid(corner=[0.0, 0.0, 0.0],a=[length, 0.0, 0.0],b=[0.0, 1.0, 0.0],c=[0.0, 0.0, hight],direction=1)
        boundaries.append(tmp_shape)
        oif.output_vtk_rhomboid(rhom_shape=tmp_shape, out_file=directory + '/wallFront.vtk')

        # back wall
        tmp_shape = shapes.Rhomboid(corner=[0.0, width - 1.0, 0.0],a=[length, 0.0, 0.0],b=[0.0, 1.0, 0.0],c=[0.0, 0.0, hight],direction=1)
        boundaries.append(tmp_shape)
        oif.output_vtk_rhomboid(rhom_shape=tmp_shape, out_file=directory + '/wallBack.vtk')

        return boundaries




def read_gro_file(file_path):
    atom_list = []
    atom_data = []
    
    with open(file_path, 'r') as gro_file:
        
        # read the header lines
        lines = gro_file.readlines()
        num_atoms = int(lines[1])
        
        # Read the atom data
        for line in lines[2:-1]:
            if len(line.strip()) == 0:
                continue
            
            atom_line = line[15:20].strip()
            atom_list.append(atom_line)
            
            residue_name = line[:10].strip()
            atom_name = str(line[10:15].strip())
            part_id = int(line[15:20].strip())
            x = float(line[20:28].strip())
            y = float(line[28:36].strip())
            z = float(line[36:44].strip())
            vx = float(line[44:53].strip())
            vy = float(line[53:62].strip())
            vz = float(line[62:71].strip())
            
            atom_data.append([residue_name, atom_name, part_id, x, y, z, vx, vy, vz])
    
    # Create a Pandas DataFrame
    columns = ['Residue', 'Atom', 'Part_id', 'X', 'Y', 'Z', 'VX', 'VY', 'VZ']
    df = pd.DataFrame(atom_data, columns=columns)
    return df, num_atoms

import espressomd
import espressomd.observables
def calc_com_velocity(system):
    obs_velcom = espressomd.observables.ComVelocity(ids= np.array(range(len(system.part))))
    velocity_com=espressomd.accumulators.TimeSeries(obs = obs_velcom)
    velocity_com.update()
    vel_temp = np.array(velocity_com.time_series())
    array = vel_temp[0,0],vel_temp[0,1],vel_temp[0,2]
    return array




def write_gro_file(atoms,  name_str_arr, filename, parameters, box = None, velocity = None):
    """
    create gro file from coordinates and velocities
    
    atoms: numpy array of shape (n_atoms, 4)
    name_str_arr: numpy array of shape (n_atom_types,)
    filename: string
    parameters: string
    box: numpy array of shape (3,)
    velocity: numpy array of shape (n_atoms, 3)
    
    """
    
    
    # extract values from atoms
    atom_type = atoms[:,0].astype(int)
    atom_coordinates = atoms[:,1:]
    
    with open(filename, 'w') as f:
        # Writing header
        f.write('Generated by Khayrul with parameters: ' + parameters + '\n')
        
        # Writing number of atoms
        n_atoms = len(atom_coordinates)
        f.write(f'{n_atoms}\n')
        
        # define atom properties
        residue_index = 1
        residue_name = 'MOL'
        
        if velocity is None:
            velocity = np.zeros((n_atoms, 3))
        
        # Writing atom coordinates
        for i, (x, y, z) in enumerate(atom_coordinates):
            
            
            # Atom name
            atom_name = name_str_arr[atom_type[i]-1]
            # Atom velocity
            vx = velocity[i, 0]
            vy = velocity[i, 1]
            vz = velocity[i, 2]
            
            # Residue index, Residue name, Atom name, Atom number, Position (in nm)
            f.write(f'{residue_index:5d}{residue_name:5s}'  \
                    f'{atom_name:5s}{i+1:5d}'  \
                    f'{x:8.3f}{y:8.3f}{z:8.3f}'  \
                    f'{vx:8.4f}{vy:8.4f}{vz:8.4f}\n')
        
        # Writing box vectors (we don't have a box here, so vectors are 0)
        if box is None:
            f.write(f'   0.00000   0.00000   0.00000\n')
        else:
            f.write(f'{box[0]:10.5f}{box[1]:10.5f}{box[2]:10.5f}\n')
            
            
            
def sim_runner(system, steps, simulation_name, name, atom_types):
    print(f"Starting {name} Ensemble:")
    system.time = 0
    part_cord_class = espressomd.observables.ParticlePositions(ids=(np.array(range(len(system.part)))))
    part_vels_class = espressomd.observables.ParticleVelocities(ids=(np.array(range(len(system.part)))))
    part_ids = np.array(range(len(system.part)))
    part_type = np.array([p.type for p in system.part])
    
    data_file = simulation_name + '/' + name + ".csv"  
    gro_folder = simulation_name + '/' + name 
    os.makedirs(gro_folder, exist_ok=True)
    
    data_array = []
    for i in range(steps):
        start_time = time.time() 
        system.integrator.run(500)
        
        # calculate center of mass velocity
        com_vel=calc_com_velocity(system)
        pot_energy = system.analysis.energy()['total'] - system.analysis.energy()['kinetic']
        data_array.append([system.time, system.analysis.energy()['total'], system.analysis.energy()['kinetic'], pot_energy, com_vel[0], com_vel[1], com_vel[2]])
        
        part_cords = part_cord_class.calculate()
        part_vels = part_vels_class.calculate()
        atoms = np.hstack((part_type.reshape(-1,1), part_cords))
        filename = gro_folder + '/' + name + '.' + str(i).zfill(5) + '.gro'
        write_gro_file(atoms,  atom_types, filename, parameters='t =' + str(system.time), box = system.box_l, velocity = part_vels)
        print(f"Step {i} took {time.time() - start_time:.2f} seconds")
        
    df = pd.DataFrame(data_array, columns = ['time', 'total_energy', 'kinetic_energy', 'potential_energy', 
                                             'COM_vx', 'COM_vy', 'COM_vz'])
    # save dataframe to csv
    df.to_csv(data_file, index=False)
    
    print(f"Finished {name} Ensemble:")