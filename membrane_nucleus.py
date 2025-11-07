# Description {{{
"""
Units:
    Basic:
        Length: µm
        Mass: g
        Time: s
    Derived:
        Mass density: g/(µm)³
        Force: nN
        Pressure: kPa
        Energy: 10^-15 J
"""
# }}}

# Libraries {{{
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, io, plot, log
from dolfinx.fem.petsc import NonlinearProblem, assemble_matrix, create_matrix
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import Constant, Function, dirichletbc, Expression, form, assemble_scalar
from dolfinx.io import XDMFFile, VTKFile
from dolfinx.la import create_petsc_vector

import ufl
from ufl import (TestFunctions, TrialFunction, Identity, grad, inner, det, div, dot, inv, tr, as_vector, outer, derivative, dev, sqrt)

import basix
from basix.ufl import element, quadrature_element

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = False
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"],
    "font.size" : 9})
plt.close("all")
from datetime import datetime
import pyvista
import gmsh
from shapely.geometry import Polygon
import copy
from pdb import set_trace

# In-house modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from python_utils.output_utils import *
from python_utils.mesh_utils import *
from python_utils.misc_utils import *
from python_utils.upLagrangian_utils import *
from python_utils.mecha_utils import *
from python_utils.solver_utils import *
from python_utils.gspde_utils_project import *
from python_utils.turnover_utils import *
# }}}

# Setting {{{
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
log.set_log_level(log.LogLevel.WARNING)

visualisation = 0
np.random.seed(16)
# }}}

# Parameters {{{
# Geometry-mesh
nucleusDia = 10.0
cellDia = 26.0
lc_ce = 5.0e-2
lc_n = 5.0e-2
meshOrder = 2
# Material
Aref_ce = (cellDia/2.0)**2.0*np.pi
periRef_ce = (cellDia/2.0)*2.0*np.pi
Aref_n = (nucleusDia/2.0)**2.0*np.pi
periRef_n = (nucleusDia/2.0)*2.0*np.pi
# Membrane tension is given here in nN/µm and bending stiffness in 10^-15 J
tensionStiffness_cell = 1.6
bendingStiffness_cell = 1.0e-4
tensionStiffness_n = tensionStiffness_cell*10
bendingStiffness_n = bendingStiffness_cell*10
# Initial conditions
opre0_ce = 0.0
opre0_n = 0.0
# Results name
results_name = "membrane_nucleus/resu" # Time scheme
Ttot = 3.5
dt = 1.0e-3 
print_each = 1
#Viscous force
omega = 1e-1 
# Repulsive force
rep_tol = 1.0e-1
rep_mag = 1.0e0
rep_st  = 5.0e0
rep_normal_tol = -0.7
# Cytoplasm force
Fc = 16 # force magnitude
# External barrier
# bmDia = 10.0
# bmHeight = 6.0
# bmCenter_num1 = (15.0, bmDia/2+bmHeight/2)  
# bmCenter1 = ufl.as_vector(bmCenter_num1)
# bmCenter_num2 = (15.0, -bmDia/2-bmHeight/2)  
# bmCenter2 = ufl.as_vector(bmCenter_num2)
# bmCenters = [bmCenter1, bmCenter2] 
# Microchannel geometry
length = 100
height = 20
width = 6
x_left = 20.0
# Barrier force
Fbarrier = 30.0
steepness = 3.5
# Mechanical forces
E_ce = 0.0103
nu_ce = 0.45
E_n = 5.0
nu_n = 0.45
eta1 = 0.02 #8
eta2 = 0.3 #30
# Nucleus to cytoplasm force
k_n = 10.0
factor = 4
# Repulsive force
alpha = 5.0
k_rep = 12.0
# Retaining force
k_t = 3.0

# Solver
quadrature_degree = 8
def SetSolverOpt(solver):
    # Newton solver
    solver.convergence_criterion = "incremental"
    solver.rtol = 1.0e-8
    solver.atol = 1.0e-8
    solver.max_it = 25
    solver.report = True
    solver.relaxation_parameter = 1.0
    # Krylov solver
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts[f"{option_prefix}ksp_type"]   = "preonly"
    opts[f"{option_prefix}pc_type"]    = "lu"
    opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
    opts[f"{option_prefix}ksp_max_it"] = 1000
    ksp.setFromOptions()
    return
# }}}

# Make mesh {{{
def MakeCircle(radius : float, lc : float, meshOrder = 1):
    model = gmsh.model
    geo = model.occ
    model.add("cell")
    # Create points
    cc = geo.addPoint(0.0, 0.0, 0.0, lc)
    rp = geo.addPoint(radius, 0.0, 0.0, lc)
    lp = geo.addPoint(-radius, 0.0, 0.0, lc)
    # Create lines
    topLine = geo.addCircleArc(rp, cc, lp)
    botLine = geo.addCircleArc(lp, cc, rp)
    # Create loop
    loop = geo.addCurveLoop([topLine, botLine])
    # Synchronise
    geo.synchronize()
    # Groups
    gr_gamma = model.addPhysicalGroup(1, [topLine, botLine])
    # Mesh
    model.mesh.generate(dim = 1)
    model.mesh.setOrder(meshOrder)
    # gmsh.fltk.run()
    return model
# }}}
# Initialisation
gmsh.initialize()
# Set cell {{{
model_ce = MakeCircle(cellDia/2.0, lc_ce, meshOrder = meshOrder)
cell_params = {
        "model" : model_ce,
        "quadrature_degree" : quadrature_degree,
        "meshOrder" : meshOrder,
        "dt" : dt,
        "opre0" : opre0_ce,
        "SetSolverOpt" : SetSolverOpt,
        "Dia" : cellDia,
        "aRef" : Aref_ce,
        "periRef" : periRef_ce,
        "Href" : 2.0/cellDia,
        "rep_mag" : rep_mag,
        "rep_st" : rep_st,
        "rep_tol" : rep_tol,
        "length" : length,
        "height" : height,
        "width" : width,
        "x_left" : x_left,
        "steepness" : steepness, 
        "omega" : omega, 
        "strength" : Fbarrier, 
        "Fc" : Fc, 
        "mech_model" : "elasticity",  
        "E" : E_ce,
        "nu" : nu_ce,
        "eta1" : eta1,
        "eta2" : eta2,
        "k_n" : k_n,
        "factor" : factor,
        "alpha" : alpha,
        "k_rep" : k_rep,
        "k_t" : k_t,
        "typeOpressure" : "area",
        "equidistribute" : True,
        "role" : "cell",
        }
cellGS = GSPDE(**cell_params)
# }}}

# Set basement membrane {{{
n_params = copy.deepcopy(cell_params)
model_n = MakeCircle(nucleusDia/2.0, lc_n, meshOrder = meshOrder)
n_params["model"] = model_n
n_params["aRef"] = Aref_n
n_params["periRef"] = periRef_n
n_params["Dia"] = nucleusDia
n_params["Href"] = 2.0/nucleusDia
n_params["opre0"] = opre0_n
n_params["E"] = E_n
n_params["nu"] = nu_n
n_params["mech_model"] = "elasticity"
#n_params["strength"] = 1e-06
n_params["Fc"] = 1e-8
n_params["typeOpressure"] = "area"
n_params["equidistribute"] = True
n_params["role"] = "nucleus"

nGS = GSPDE(other_gspde=cellGS, **n_params)
cellGS.other_gspde = nGS
# }}}

# Set properties {{{
# Cell bending and tension stiffness
cellGS.tensionStiffness.x.array[:] = tensionStiffness_cell
cellGS.bendingStiffness.x.array[:] = bendingStiffness_cell
# Basement membrane bending and tension
nGS.tensionStiffness.x.array[:] = tensionStiffness_n
nGS.bendingStiffness.x.array[:] = bendingStiffness_n

surface_energy_form = form(cellGS.tensionStiffness * cellGS.dx) 
surface_energy = assemble_scalar(surface_energy_form)
SurfaceEnergyList = [surface_energy]

bending_energy_form = form(cellGS.bendingStiffness/2 * cellGS.H_old**2 * cellGS.dx) 
bending_energy = assemble_scalar(bending_energy_form)
BendingEnergyList = [bending_energy]

# Set up output {{{
# Cell
out_ce = Output(cellGS.domain, [cellGS.disp, cellGS.H_old, cellGS.normal,
                                cellGS.selfRepuForce, cellGS.mechForce, cellGS.barrierForce, cellGS.movForce, cellGS.repulsiveForce, cellGS.retainForce, cellGS.phi, cellGS.bendingStiffness, cellGS.tensionStiffness],
                ["u", "H", "n", "Fsr", "Fm", "Fbar", "Fc", "Frep", "Fret", "phi", "Fb", "Fs"],
                results_name + "_ce", comm)
timeList = [0.0]
areaList_ce = [cellGS.area]
periList_ce = [cellGS.perimeter]
velocityList_ce = [0.0]
stressList_ce = [0.0]
x_front = [cellDia/2]

# Basement membrane
out_n = Output(nGS.domain, [nGS.disp, nGS.H_old, nGS.normal,
                              nGS.selfRepuForce, nGS.mechForce, nGS.barrierForce, nGS.nucleusForce, nGS.repulsiveForce, nGS.phi, nGS.bendingStiffness, nGS.tensionStiffness],
                ["u", "H", "n", "Fsr", "Fm", "Fb", "Fn", "Frep", "phi", "Fb", "Fs"],
                results_name + "_n", comm)
areaList_n = [nGS.area]
periList_n = [nGS.perimeter]
x_center_old = compute_center(cellGS.domain)
# }}}

# # Plot the barrier {{{
# # Initialize gmsh and create geometry
# model = gmsh.model
# geo = model.occ
# model.add("barrier")
# lc = 0.5 
# # First circle
# cp1 = geo.addPoint(*bmCenter_num1, 0.0, lc)
# rp1 = geo.addPoint(bmCenter_num1[0] + bmDia/2, bmCenter_num1[1], 0.0, lc)
# lp1 = geo.addPoint(bmCenter_num1[0] - bmDia/2, bmCenter_num1[1], 0.0, lc)
# arc1a = geo.addCircleArc(rp1, cp1, lp1)
# arc1b = geo.addCircleArc(lp1, cp1, rp1)
# loop1 = geo.addCurveLoop([arc1a, arc1b])
# surface1 = geo.addPlaneSurface([loop1])
# # Second circle
# cp2 = geo.addPoint(*bmCenter_num2, 0.0, lc)
# rp2 = geo.addPoint(bmCenter_num2[0] + bmDia/2, bmCenter_num2[1], 0.0, lc)
# lp2 = geo.addPoint(bmCenter_num2[0] - bmDia/2, bmCenter_num2[1], 0.0, lc)
# arc2a = geo.addCircleArc(rp2, cp2, lp2)
# arc2b = geo.addCircleArc(lp2, cp2, rp2)
# loop2 = geo.addCurveLoop([arc2a, arc2b])
# surface2 = geo.addPlaneSurface([loop2])

# geo.synchronize()
# model.addPhysicalGroup(1, [arc1a, arc1b, arc2a, arc2b], tag=1)
# model.setPhysicalName(1, 1, "barrier_arcs")
# model.addPhysicalGroup(2, [surface1, surface2], tag=2)
# model.setPhysicalName(2, 2, "barrier_surfaces")

# model.mesh.generate(2)
# gmsh.write("results/vtk/membrane_nucleus/barrier_plot.vtk")
# gmsh.finalize()
# # }}}

# # Plot the barrier {{{
gmsh.model.add("barrier")
geo = gmsh.model.occ

lc = 0.5
centres = [
    (x_left, +width/2 + height / 2.0),    # top-left
    (x_left, -width/2 - height / 2.0),    # bottom-left
    (x_left + length, +width/2 + height / 2.0),  # top-right
    (x_left + length, -width/2 - height / 2.0)   # bottom-right
]

surfaces = []

for i, (cx, cy) in enumerate(centres):
    center = geo.addPoint(cx, cy, 0, lc)

    if i < 2:  # semicerchi sinistri (da 90° a 270°)
        pt_top = geo.addPoint(cx, cy + height / 2.0, 0, lc)  # 90°
        pt_mid = geo.addPoint(cx - height / 2.0, cy, 0, lc)  # 180°
        pt_bot = geo.addPoint(cx, cy - height / 2.0, 0, lc)  # 270°

        arc1 = geo.addCircleArc(pt_top, center, pt_mid)
        arc2 = geo.addCircleArc(pt_mid, center, pt_bot)
        line = geo.addLine(pt_bot, pt_top)
    else:  # semicerchi destri (da -90° a 90°)
        pt_bot = geo.addPoint(cx, cy - height / 2.0, 0, lc)  # -90°
        pt_mid = geo.addPoint(cx + height / 2.0, cy, 0, lc)  # 0°
        pt_top = geo.addPoint(cx, cy + height / 2.0, 0, lc)  # 90°

        arc1 = geo.addCircleArc(pt_bot, center, pt_mid)
        arc2 = geo.addCircleArc(pt_mid, center, pt_top)
        line = geo.addLine(pt_top, pt_bot)

    loop = geo.addCurveLoop([arc1, arc2, line])
    surf = geo.addPlaneSurface([loop])
    surfaces.append(surf)

# Parete superiore
x0 = x_left
x1 = x_left + length
y_top = width / 2
y_bot = -width / 2

p1 = geo.addPoint(x0, y_top, 0, lc)
p2 = geo.addPoint(x1, y_top, 0, lc)
p3 = geo.addPoint(x1, y_top + height, 0, lc)
p4 = geo.addPoint(x0, y_top + height, 0, lc)
l1 = geo.addLine(p1, p2)
l2 = geo.addLine(p2, p3)
l3 = geo.addLine(p3, p4)
l4 = geo.addLine(p4, p1)
loop_top = geo.addCurveLoop([l1, l2, l3, l4])
surf_top = geo.addPlaneSurface([loop_top])
surfaces.append(surf_top)

# Parete inferiore
p5 = geo.addPoint(x0, y_bot, 0, lc)
p6 = geo.addPoint(x1, y_bot, 0, lc)
p7 = geo.addPoint(x1, y_bot - height, 0, lc)
p8 = geo.addPoint(x0, y_bot - height, 0, lc)
l5 = geo.addLine(p5, p6)
l6 = geo.addLine(p6, p7)
l7 = geo.addLine(p7, p8)
l8 = geo.addLine(p8, p5)
loop_bot = geo.addCurveLoop([l5, l6, l7, l8])
surf_bot = geo.addPlaneSurface([loop_bot])
surfaces.append(surf_bot)

geo.synchronize()

# Aggiungi Physical Group per le superfici
gmsh.model.addPhysicalGroup(2, surfaces, 1)
gmsh.model.setPhysicalName(2, 1, "barrier_surfaces")

gmsh.model.mesh.generate(2)
os.makedirs("results/vtk/membrane_nucleus", exist_ok=True)
gmsh.write("results/vtk/membrane_nucleus/barrier_plot.vtk")
gmsh.finalize()

# # }}}

# Calculation loop {{{
# Initialisation
t = 0.0
out_ce.WriteResults(t = t)
out_n.WriteResults(t = t)
mprint("------------------------------------", rank = rank)
mprint("Simulation Start", rank = rank)
mprint("------------------------------------", rank = rank)
startTime = datetime.now()
printTime0 = datetime.now()
# To solve variables
gspdes_list = [cellGS, nGS]
toSolve_list = [0, 1] # Solve cellGS and nGS
# Time stepping solution procedure loop
k1 = 0

while (round(t + dt, 9) <= Ttot):
    # Update iteration
    k1 += 1
    # Solution
    t += dt
    SolveIteration(k1, t, gspdes_list, toSolve_list)

    # AdaptiveTimeSolver(k1, t, dt, rep_mag/10.0, gspdes_list, toSolve_list, [0])
    # Report area and perimeter
    timeList.append(t)
    areaList_ce.append(cellGS.area)
    periList_ce.append(cellGS.perimeter)
    areaList_n.append(nGS.area)
    periList_n.append(nGS.perimeter)
     # Save velocity 
    x_center = compute_center(cellGS.domain)
    vel_norm = np.linalg.norm((x_center - x_center_old) / dt)
    velocityList_ce.append(vel_norm)
    # Save stress
    stressList_ce.append(cellGS.avg_sigma_n)
    x_center_old = x_center
    # Save front
    x_front.append(cellGS.x_front)
    # Save surface energy
    surface_energy_form = form(cellGS.tensionStiffness * cellGS.dx) 
    surface_energy = assemble_scalar(surface_energy_form)
    SurfaceEnergyList.append(surface_energy)
    # Save bending energy
    bending_energy_form = form(cellGS.bendingStiffness/2 * cellGS.H_old**2 * cellGS.dx) 
    bending_energy = assemble_scalar(bending_energy_form)
    BendingEnergyList.append(bending_energy)
    # Print progress
    printTime1 = datetime.now()
    cpu_time = printTime1 - printTime0
    printTime0 = printTime1
    mprint("------------------------------------", rank = rank)
    mprint("Increment: {} | CPU time: {}".format(k1, cpu_time), rank = rank)
    mprint("dt: {} s | Simulation time {} s of {} s".format(round(dt, 4), round(t, 4), Ttot), rank = rank)
    mprint("", rank = rank)
    mprint("------------------------------------", rank = rank)
    # Write output results
    if k1%print_each == 0:
        out_ce.WriteResults(t)
        out_n.WriteResults(t)
        # Save data file
        data = {
            "Time" : np.array(timeList),
            "Area_ce" : np.array(areaList_ce),
            "Area_n" : np.array(areaList_n),
            "Peri_ce" : np.array(periList_ce),
            "Peri_n" : np.array(periList_n),
            "Vel_ce" : np.array(velocityList_ce),  
            "Stress_ce" : np.array(stressList_ce),
            "x_front" : np.array(x_front),
            "surface_energy" : np.array(SurfaceEnergyList),
            "bending_energy" : np.array(BendingEnergyList)
        }
        data = pd.DataFrame(data)
        data.to_csv("results/" + results_name + ".csv")
# Close files
out_ce.Close()
out_n.Close()

mprint("-----------------------------------------", rank = rank)
mprint("End computation", rank = rank)
# Report elapsed real time for the analysis
endTime = datetime.now()
elapseTime = endTime - startTime
mprint("------------------------------------------", rank = rank)
mprint("Elapsed real time:  {}".format(elapseTime))
mprint("------------------------------------------", rank = rank)

