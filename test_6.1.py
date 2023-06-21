#!/usr/bin/python3
# squizeeing a lipid vesicle
## Dia=60
##
import espressomd
required_features = ["LB_BOUNDARIES","EXTERNAL_FORCES", "SOFT_SPHERE", "LENNARD_JONES", "MASS","LJCOS2"]
espressomd.assert_features(required_features)
from espressomd import lbboundaries
from espressomd import shapes
from espressomd.interactions import FeneBond
from espressomd.interactions import AngleHarmonic
from espressomd.io.writer import vtf
import espressomd.accumulators
import espressomd.observables
import object_in_fluid as oif
from object_in_fluid.oif_utils import output_vtk_rhomboid, output_vtk_cylinder
import numpy as np
import os,shutil,glob, subprocess
from os import system, name
import sys,warnings,logging,time
import espressomd.visualization
import threading

# import matplotlib.pyplot as pyplot
from espressomd import checkpointing
# from matplotlib import rc
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure
# from PIL import Image
from io import BytesIO
import MDAnalysis as mda
from espressomd import MDA_ESP
from MDAnalysis.coordinates.TRR import TRRWriter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")

# simulation settings

simulation_name = 'test_6.1'
if os.path.exists(simulation_name) and os.path.isdir(simulation_name):
    shutil.rmtree(simulation_name)
os.mkdir(simulation_name)
os.chdir (simulation_name)
os.mkdir ("gro_file")
os.mkdir ("gro_file/trajectory")
gro_bash_file = open("gro_bash.sh", "w")

matplotlib_notebook = False    # toggle this off when outside IPython/Jupyter
live_plotting = False
copy_file = "test_6.1.py"
shutil.copy('../'+copy_file, copy_file)

# system parameters

D=60
d=1
BOX_L = 6*D
BOX_W = 2*D
BOX_H = 2*D
compression_w = 0.5
compression_l = D*2

# initialization
periodicity = [True,True,True]
time_step = 0.01
system = espressomd.System(box_l=(BOX_L, BOX_W, BOX_H))
#system.seed = 42
system.time_step = time_step
system.cell_system.skin = 1.0

# Function and classes


class geometry:
    directory = 'geometry_file'

    def __init__(self,dia,flag):
        self.dia = dia
        if flag:
            os.mkdir(geometry.directory)
    def rectangle(self,BOX_L,BOX_W,BOX_H,compression_w,compression_l):
        length  = BOX_L
        width  = BOX_W
        hight  = BOX_H
        sq_hight = BOX_W*compression_w*0.5
        sq_length = compression_l

        boundaries=[]
        # bottom wall
        tmp_shape = shapes.Rhomboid(corner=[0.0, 0.0, 0.0],a=[length, 0.0, 0.0],b=[0.0, width, 0.0],c=[0.0, 0.0, 1.0], direction=1)
        boundaries.append(tmp_shape)
        oif.output_vtk_rhomboid(rhom_shape=tmp_shape, out_file=geometry.directory + '/wallBottom.vtk')

        # top wall
        tmp_shape = shapes.Rhomboid(corner=[0.0, 0.0, hight],a=[length, 0.0, 0.0],b=[0.0, width, 0.0],c=[0.0, 0.0, - 1.0], direction=1)
        boundaries.append(tmp_shape)
        oif.output_vtk_rhomboid(rhom_shape=tmp_shape, out_file=geometry.directory + '/wallTop.vtk')

        # front wall
        tmp_shape = shapes.Rhomboid(corner=[0.0, 0.0, 0.0],a=[length, 0.0, 0.0],b=[0.0, 1.0, 0.0],c=[0.0, 0.0, hight],direction=1)
        boundaries.append(tmp_shape)
        oif.output_vtk_rhomboid(rhom_shape=tmp_shape, out_file=geometry.directory + '/wallFront.vtk')

        # back wall
        tmp_shape = shapes.Rhomboid(corner=[0.0, width - 1.0, 0.0],a=[length, 0.0, 0.0],b=[0.0, 1.0, 0.0],c=[0.0, 0.0, hight],direction=1)
        boundaries.append(tmp_shape)
        oif.output_vtk_rhomboid(rhom_shape=tmp_shape, out_file=geometry.directory + '/wallBack.vtk')

        # bottom rectangle
        tmp_shape = shapes.Rhomboid(corner=[(length-sq_length)/2, 0.0, 1],a=[sq_length, 0.0, 0.0],b=[0.0, width, 0.0],c=[0.0, 0.0, sq_hight],direction=1)
        boundaries.append(tmp_shape)
        oif.output_vtk_rhomboid(rhom_shape=tmp_shape, out_file=geometry.directory + '/BottomRectangle.vtk')

        #top rectangle
        tmp_shape = shapes.Rhomboid(corner=[(length-sq_length)/2, 0.0, hight-1],a=[sq_length, 0.0, 0.0],b=[0.0, width, 0.0],c=[0.0, 0.0, - sq_hight],direction=1)
        boundaries.append(tmp_shape)
        oif.output_vtk_rhomboid(rhom_shape=tmp_shape, out_file=geometry.directory + '/TopRectangle.vtk')

        # add constraints
        system.lbboundaries.clear()
        # creating boundaries
        for boundary in boundaries:
            system.lbboundaries.add(lbboundaries.LBBoundary(shape=boundary, velocity=[0.0,0.0,0.0]))
            system.constraints.add(shape=boundary, particle_type=2, penetrable=False)

        system.non_bonded_inter[0, 2].lennard_jones.set_params(epsilon=1.0,
                                                       sigma=1.0,
                                                       cutoff=1.12246204831,
                                                       shift = 'auto')

        system.non_bonded_inter[1, 2].lennard_jones.set_params(epsilon=1.0,
                                                       sigma=1.0,
                                                       cutoff=1.12246204831,
                                                       shift = 'auto')

    def vesicle(self,d,BOX_L,BOX_W,BOX_H):
        CenterX = self.dia
        CenterY = np.floor (BOX_W/2.0)
        CenterZ = np.floor (BOX_H/2.0)
        # Vesicle Diameter
        R = D/2.0
        # Diameter of vesicle is the diamter of sphere between two lipid HTT|TTH
        R = R-11*d/2.0
        # Particle Diameter
        section_number = np.floor(np.pi/(2.0*np.arcsin(d/(2*R))))
        size = int(section_number+2)
        X=np.zeros((size))
        Y=np.zeros((size))
        Z=np.zeros((size))
        points=[]
        points = np.array(points)
        first_point=np.array([0,0,R])
        second_point=[0,0,-R]
        second_point
        # First Points
        points=np.append(points,first_point)
        points=np.append([points],[second_point],axis=0)
        n=1
        for i in range(2,size):
            ti=(i-1)*np.pi/np.floor(np.pi/(2.0*np.arcsin(d/(2*R))))
            for j in range(1,1+int(np.floor(np.pi*np.sin(ti)/np.arcsin(d/(2*R))))):
                n=n+1
                fj = (j-1)*2.0*np.pi/(np.floor(np.pi*np.sin(ti)/np.arcsin(d/(2*R))))
                X=R*np.sin(ti)*np.cos(fj)
                Y=R*np.sin(ti)*np.sin(fj)
                Z=R*np.cos(ti)
                new_point = np.array([X,Y,Z])
                new_point.shape
                points=np.append(points,[new_point],axis=0)
        x= list(points[:,0])
        y= list(points[:,1])
        z= list(points[:,2])
        av= np.linalg.norm(points,axis=1)
        av=np.transpose(np.array([av]*3))
        norm = points/av
        bond_length = np.average(np.linalg.norm(norm,axis=1))
        print(bond_length)
        second_points = points+1*norm
        third_points  = points+2*norm
        fourth_points = points+3*norm +2*norm
        fifth_points  = points+4*norm +2*norm
        sixth_points  = points+5*norm +2*norm
        print('hi')
        fp = open(geometry.directory + '/lipid.xyz', mode='w')
        fp.write(str(6*len(points))+"\n")
        fp.write("This file is made by python\n")
        for i in range (0, len(points)):
            fp.write("{}\t{}\t{}\t{}\t\n".format(1,points[i,0],points[i,1],points[i,2]))
            fp.write("{}\t{}\t{}\t{}\t\n".format(2,second_points[i,0],second_points[i,1],second_points[i,2]))
            fp.write("{}\t{}\t{}\t{}\t\n".format(3,third_points[i,0],third_points[i,1],third_points[i,2]))
            fp.write("{}\t{}\t{}\t{}\t\n".format(1,sixth_points[i,0],sixth_points[i,1],sixth_points[i,2]))
            fp.write("{}\t{}\t{}\t{}\t\n".format(2,fifth_points[i,0],fifth_points[i,1],fifth_points[i,2]))
            fp.write("{}\t{}\t{}\t{}\t\n".format(3,fourth_points[i,0],fourth_points[i,1],fourth_points[i,2]))
        fp.close()

        # HEAD-TAIL MODEL IMPLEMENTATION
        # ############################################################################################################
        f = open(geometry.directory + '/lipid.xyz',  'r')
        N_PART = int(f.readline())
        data = np.loadtxt (geometry.directory + '/lipid.xyz', dtype = float, skiprows = 2)
        num = 0
        # for i in range(0, int(n/3.0)):
        #     system.part.add(id = 3*i,   pos=[data[3*i,1]  +CenterX, data[3*i,2]  +CenterY, data[3*i,3]  +CenterZ], type=0)
        #     system.part.add(id = 3*i+1, pos=[data[3*i+1,1]+CenterX, data[3*i+1,2]+CenterY, data[3*i+1,3]+CenterZ], type=1)
        #     system.part.add(id = 3*i+2, pos=[data[3*i+2,1]+CenterX, data[3*i+2,2]+CenterY, data[3*i+2,3]+CenterZ], type=1)
        for i in range(0, int(N_PART/6.0)):
            system.part.add(id = 6*i,   pos=[data[6*i,  1]+CenterX, data[6*i,  2]+CenterY, data[6*i,  3]+CenterZ], type= 0)  #11
            system.part.add(id = 6*i+1, pos=[data[6*i+1,1]+CenterX, data[6*i+1,2]+CenterY, data[6*i+1,3]+CenterZ], type= 1)  #21
            system.part.add(id = 6*i+2, pos=[data[6*i+2,1]+CenterX, data[6*i+2,2]+CenterY, data[6*i+2,3]+CenterZ], type= 2)  #22
            system.part.add(id = 6*i+3, pos=[data[6*i+3,1]+CenterX, data[6*i+3,2]+CenterY, data[6*i+3,3]+CenterZ], type= 3)  #12
            system.part.add(id = 6*i+4, pos=[data[6*i+4,1]+CenterX, data[6*i+4,2]+CenterY, data[6*i+4,3]+CenterZ], type= 4)  #23
            system.part.add(id = 6*i+5, pos=[data[6*i+5,1]+CenterX, data[6*i+5,2]+CenterY, data[6*i+5,3]+CenterZ], type= 5)  #24
        # creating the universie
        eos = MDA_ESP.Stream(system)  # create the stream
        u = mda.Universe(eos.topology, eos.trajectory)  # create the MDA universe
        u.atoms.write("my_lipid.gro")
        with open("my_lipid.gro") as f:
            lines = f.readlines()
        lines[0] = f"Gro file by Khayrul,t={0.0} \n"
        with open("my_lipid.gro","w") as f:
            f.writelines(lines)
        gro_bash_file = open("gro_bash.sh", "a")
        gro_bash_file.write("gmx_mpi trjconv -f my_lipid.gro -o my_lipid.trr \n")
        gro_bash_file.close()

        # NON_BONDED INTERACTIONS
        # #############################################################################################################
        # system.non_bonded_inter[0, 0].lennard_jones.set_params(epsilon=1.0, sigma=0.95, cutoff=0.95*1.12246204831, shift =0.25)
        # system.non_bonded_inter[0, 1].lennard_jones.set_params(epsilon=1.0, sigma=0.95, cutoff=0.95*1.12246204831, shift =0.25)
        # system.non_bonded_inter[1, 1].lennard_jones_cos2.set_params(epsilon=1.0, sigma=1.0 , width=1.6)
        #particle_types=[11,21,22,12,23,24]
        particle_types=[0,1,2,3,4,5]
        tail_types=[1,2,4,5]
        for i,i_elem in enumerate(particle_types):
            for j,j_elem in enumerate(particle_types):
                if i==j and i not in tail_types:
                    system.non_bonded_inter[i_elem, j_elem].lennard_jones.set_params(epsilon=1.0, sigma=0.95, cutoff=0.95*2**(1./ 6), shift =0.25)
                    print(f"defining non-bonded interaction between {i_elem} and {j_elem}")
                elif j>i:
                    # if i in tail_types and j in tail_types:
                    #     system.non_bonded_inter[i_elem, j_elem].lennard_jones.set_params(epsilon=1.0, sigma=1.0, cutoff=1*2**(1./ 6), shift =0.25)
                    #     print(f"defining non-bonded tail interaction between {i_elem} and {j_elem}")
                    #     continue
                    system.non_bonded_inter[i_elem, j_elem].lennard_jones.set_params(epsilon=1.0, sigma=0.95, cutoff=0.95*2**(1./ 6), shift =0.25)
                    print(f"defining non-bonded interaction between {i_elem} and {j_elem}")
        #tail_types=[21,22,23,24]

        for i,i_elem in enumerate(tail_types):
            for j,j_elem in enumerate(tail_types):
                if i==j:
                    system.non_bonded_inter[i_elem, j_elem].lennard_jones_cos2.set_params(epsilon=1.0, sigma=1.0, width=1.6)
                if j>i:
                    system.non_bonded_inter[i_elem, j_elem].lennard_jones_cos2.set_params(epsilon=1.0, sigma=1.0, width=1.6)
                print(f"defining lj_cos2 interaction between {i_elem} and {j_elem}")

        # BONDED INTERACTIONS
        # #############################################################################################################
        fene = FeneBond(k=30, d_r_max=1.5,r_0=0.0)
        system.bonded_inter.add(fene)
        for i in range(0, int(N_PART/3.0)):
            system.part[3*i+1].add_bond((fene, 3*i))
            system.part[3*i+1].add_bond((fene, 3*i+2))

        angle_harmonic = AngleHarmonic(bend=10.0, phi0= np.pi)
        system.bonded_inter.add(angle_harmonic)
        for i in range(0, int(n/3.0)):
            system.part[3*i+1].add_bond((angle_harmonic,3*i, 3*i+2))

def calc_com_velocity(step,array):
    obs_velcom = espressomd.observables.ComVelocity(ids= np.array(range(len(system.part))))
    velocity_com=espressomd.accumulators.TimeSeries(obs = obs_velcom)
    velocity_com.update()
    vel_temp = np.array(velocity_com.time_series())
    array = vel_temp[0,0],vel_temp[0,1],vel_temp[0,2]
    return array
def plot_graph(graph_name,file,x_col,y_col):
    csv_data=pd.read_csv(file, sep='\t',header = 0 )
    xvg_mat = np.asarray(csv_data)
    headings=list(csv_data.columns)
    x_axis = xvg_mat[:,x_col]
    y_axis = xvg_mat[:,y_col]
    plt.figure(figsize=(16,10), dpi= 200)
    plt.plot(x_axis,y_axis,linestyle = '-',label='_Hidden label',linewidth=3.0,alpha=1)
    plt.xlabel(headings[x_col],fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel(headings[y_col],fontsize=20)
    image_format = 'svg' # e.g .png, .svg, etc.
    image_name = "{}.svg".format(graph_name)
    plt.savefig(image_name, format=image_format, dpi=200,bbox_inches='tight',transparent=True)


def main(NAME,maxcycle,steps,vsp,vsf,write_velocity,Equil):
    print(NAME+" simulation starts now\n")
    system.time = 0  # reset system timer
    #energies = np.zeros((maxcycle, 2))   # monitor system energies
    #kinetic_temperature = np.zeros((maxcycle, 2))  # monitor system temperatures
    com_velocity = np.zeros((maxcycle, 4))
    # Initializing plots and files
    fp = open(NAME+'.vtf', mode='w+t')
    vtf.writevsf(system, fp)
    vtf.writevcf(system, fp)
    # code for dumping xyz file
    eos = MDA_ESP.Stream(system)  # create the stream
    u = mda.Universe(eos.topology, eos.trajectory)  # create the MDA universe
    N_PART = len(system.part[:])
    for i in range(1,maxcycle+1):
        start_time = time.time()
        system.integrator.run(steps)
        if opengl_visualize:
            visualizer.update()
            print("updating visualization in opengl ...................")
        array=calc_com_velocity(i,com_velocity)
        if Equil=='NVT_eq':
            data_filename = "NVT_Energy_Eq.dat"
        elif Equil=='langevin_eq':
            data_filename = "Langevin_Energy_Eq.dat"
        elif Equil=='LB_eq':
            data_filename = "LB_Energy_Eq.dat"
        else:
            data_filename = "LB_prod.dat"
        with  open(data_filename, "a") as en_file:
              Total_energy = system.analysis.energy()['total']
              Kinetic_energy = system.analysis.energy()['kinetic']
              calc_temperature = 2. / 3. * Kinetic_energy / N_PART
              Potential_energy = Total_energy - Kinetic_energy
              en_file.write(str(system.time)+"\t"+str(calc_temperature) +"\t"+str(Total_energy) +"\t"+
              str(Kinetic_energy)+"\t"+str(Potential_energy)+"\t"+str(array[0])+"\t"+str(array[1])+"\t"
              +str(array[2])+"\n")
        if min(system.part[:].pos[0]) >(BOX_L+compression_l)/2:
            print(f"Stopping squeezing as it crossed the channel with position: {min(system.part[:].pos[0])}")
            break
            print(min(system.part[:].pos[0]))
        if (i%vsp==0):
            end_time = time.time()
            simu_time = end_time - start_time
            print("{} th loop time: {}".format(i,simu_time))
            system.part.writevtk(NAME+"_p_H_" + str(i) + ".vtk", types=[0])
            system.part.writevtk(NAME+"_p_T_" + str(i) + ".vtk", types=[1])
            vtf.writevcf(system, fp)
            u.load_new(eos.trajectory)
            gro_filename='../gro_file/{0}_liposome_{1:05d}.gro'.format(NAME,i)
            u.atoms.write(gro_filename)
            simu_time = system.time
            print(simu_time)
            calc_time = time_step*i
            print(calc_time)
            with open(gro_filename) as f:
                lines = f.readlines()
            lines[0] = f"Gro file by Khayrul,t={simu_time:.2f} \n"
            with open(gro_filename,"w") as f:
                f.writelines(lines)
            print("===> {} th configuration has been written on {}".format(i,gro_filename))
            with open("../gro_bash.sh", "a") as gro_bash_file:
                gro_bash_file.write(f"gmx_mpi trjconv -f gro_file/{NAME}_liposome_{i:05d}.gro -o gro_file/trajectory/{NAME}_liposome_{i:05d}.trr \n")


        if (write_velocity):
            if (i%vsf==0):
                file_name_velocity= "flow_" +NAME+ str(i) + ".vtk"
                lbf.print_vtk_velocity( file_name_velocity )

    if matplotlib_notebook:
        display.clear_output(wait=True)
    Pot_Energy_plot = plot_graph(NAME+"_potential_energy_plot",data_filename,0,4)
    tot_Energy_plot = plot_graph(NAME+"_Total_energy_plot",data_filename,0,2)
    temp_plot       = plot_graph(NAME+"_Temperature_plot",data_filename,0,1)
    return None

# ACTUAL SIMULATION
#############################################################################################################

# create vesicle geometry
geometry(D,1).vesicle(d,BOX_L,BOX_W,BOX_H)
# creating rectangle bodundary domain
geometry(D,0).rectangle(BOX_L,BOX_W,BOX_H,compression_w,compression_l)

opengl_visualize = False
if opengl_visualize:
    print("visualizing in opengl ...................")
    from espressomd import visualization
    from threading import Thread
    visualizer = espressomd.visualization.openGLLive(system)
#############################################################################################################
#
#                                   ENERGY MINIMIZATION
#
#############################################################################################################

minimization = True
if (minimization):
    system.time_step = 0.001
    F_TOL = 1e-2
    DAMPING = 20
    MAX_STEPS = 10000
    MAX_DISPLACEMENT = 0.01 * 1 #0.01 * LJ_SIG
    EM_STEP = 100
    # Set up steepest descent integration
    print(f"Energy minimization starting with minimum distance: {system.analysis.min_dist()}")
    system.integrator.set_steepest_descent(f_max=0,  # use a relative convergence criterion only
                                       gamma=DAMPING,
                                       max_displacement=MAX_DISPLACEMENT)

    # Initialize integrator to obtain initial forces
    system.integrator.run(0)
    old_force = np.max(np.linalg.norm(system.part[:].f, axis=1))
    while system.time / system.time_step < MAX_STEPS:
        system.integrator.run(EM_STEP)
        force = np.max(np.linalg.norm(system.part[:].f, axis=1))
        rel_force = np.abs((force - old_force) / old_force)
        print(f'rel. force change: {rel_force:.2e}')
        if rel_force < F_TOL:
            break
        old_force = force
    # reset clock
    system.time = 0.
    print(f"Energy minimization finished with minimum distance: {system.analysis.min_dist()}")

#############################################################################################################
#
#                                   NVT Ensemble
#
#############################################################################################################

# Parameters for the Langevin thermostat
# reduced temperature T* = k_B T / LJ_EPS
TEMPERATURE = 0.827  # value from Tab. 1 in [7]
GAMMA = 1.0
system.integrator.set_vv()
#system.thermostat.set_langevin(kT=TEMPERATURE, gamma=GAMMA, seed=42)
system.thermostat.set_langevin(kT=1.0, gamma=.5,seed=143)
# Integration parameters


# Setting-up Thermosttat
print("Starting NVT Ensemble:")
system.time_step = 0.01
system.thermostat.turn_off()
system.integrator.set_vv() # same as set_nvt
system.thermostat.set_langevin(kT=0.6, gamma=2,seed=143)

# Simulation run paramters for vesicle
NAME = "Langevin"              # should be a string; name of the current simulation
vsp = 10                       # Visualiztion steps for head-tail particles
vsf =1000                      # Visulization steps for flow
maxcycle = 500 #50
steps = 100
write_velocity=False
Equil='langevin_eq'
os.mkdir('Energy_Eq')
os.chdir ('Energy_Eq')
en_file = open("Langevin_Energy_Eq.dat", "w")

en_file.write("time \t temperature \t Total_energy \t Kinetic_energy\t Potential_energy\t COM_velocity_x \t COM_velocity_y \t COM_velocity_z \n")
en_file.close()
# Simulation for vesicle
main(NAME,maxcycle,steps,vsp,vsf,write_velocity,Equil)





#main(NAME,maxcycle,steps,vsp,vsf,write_velocity,Equil)
with open("../gro_bash.sh", "a") as gro_bash_file:
    gro_bash_file.write(f"gmx_mpi trjcat -f gro_file/trajectory/{NAME}_liposome_*.trr -o {NAME}_combined.trr \n")


# system.galilei.galilei_transform Subtracts the velocity of the center
# of mass of the whole system from every particle’s velocity
system.galilei.galilei_transform()
system.thermostat.turn_off()

#############################################################################################################
#
#                                   LBM Equilibration
#
#############################################################################################################

system.time_step = 0.0001

# Simulation run paramters for LBM
NAME = "LB_Eq"
vsp = 10
vsf = 100
maxcycle =1000 #500
steps = 100
write_velocity=True
Equil="LB_eq"


# creating text files and folders

en_file = open("LB_Energy_Eq.dat", "w")
en_file.write("time \t temperature \t Total_energy \t Kinetic_energy \t Potential_energy\t COM_velocity_x \t COM_velocity_y \t COM_velocity_z \n")
en_file.close()

lbf = espressomd.lb.LBFluidGPU(kT=1.0,
                               seed=123,
                               agrid=1,
                               dens=1,
                               visc=1.4267,
                               tau=0.01,
                               ext_force_density=[0.0, 0.0, 0.0])

# minimaization run
system.actors.clear()
system.actors.add(lbf)
system.thermostat.set_lb(LB_fluid=lbf, seed=142, gamma=.20)
main(NAME,maxcycle,steps,vsp,vsf,write_velocity,Equil)
with open("../gro_bash.sh", "a") as gro_bash_file:
    gro_bash_file.write(f"gmx_mpi trjcat -f gro_file/trajectory/{NAME}_liposome_*.trr -o {NAME}_combined.trr \n")

os.chdir ('..')
os.mkdir('Production_run')
os.chdir ('Production_run')
en_file = open("LB_prod.dat", "w")
en_file.write("time \t temperature \t Total_energy \t Kinetic_energy \t Potential_energy\t COM_velocity_x \t COM_velocity_y \t COM_velocity_z \n")
en_file.close()

lbf = espressomd.lb.LBFluidGPU(kT=1.0,
                               seed=123,
                               agrid=1,
                               dens=1,
                               visc=1.4267,
                               tau=0.01,
                               ext_force_density=[0.04, 0.0, 0.0])

#############################################################################################################
#
#                                   Production
#
#############################################################################################################
vsp = 10
vsf = 10
maxcycle =50000
steps = 100
system.time_step = 0.01
Equil="LB_prod"
NAME = "LB"
system.actors.clear()
system.actors.add(lbf)
system.thermostat.set_lb(LB_fluid=lbf, seed=142, gamma=.2)
main(NAME,maxcycle,steps,vsp,vsf,write_velocity,Equil)
with open("../gro_bash.sh", "a") as gro_bash_file:
    gro_bash_file.write(f"gmx_mpi trjcat -f gro_file/trajectory/{NAME}_liposome_*.trr -o {NAME}_combined.trr \n")

os.chdir ("..")
