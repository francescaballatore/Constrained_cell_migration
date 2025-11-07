# Libraries {{{
import dolfinx
from dolfinx import plot, fem
import pyvista

import ufl
from ufl import (TestFunction, TrialFunction, Identity, grad, inner, det,
                 inv, tr, as_vector, outer, derivative, dev, sqrt)

import numpy as np

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Helvetica"],
    "font.size" : 9})
plt.close("all")
# }}}

# Visualisation {{{
def Visu(domain, vdim, tagObject, tags):
    # Set up plotter
    plotter = pyvista.Plotter()
    plotter.add_text("Mesh", font_size = 14, color = "black",
                     position = "upper_edge")
    # Add whole mesh
    unGrid = pyvista.UnstructuredGrid
    plotter.add_mesh(unGrid(*plot.vtk_mesh(domain, domain.topology.dim)),
                     show_edges = True, show_scalar_bar = False)
    # Add sub meshes: cells
    if vdim == 1:
        for tag_i in tags:
            plotter.add_mesh(unGrid(*plot.vtk_mesh(domain,
                                                   entities = tagObject.find(tag_i),
                                                   dim = vdim)),
                             show_edges = True,
                             edge_color = "black",
                             line_width = 10)
    else:
        for tag_i in tags:
            plotter.add_mesh(unGrid(*plot.vtk_mesh(domain,
                                                   entities = tagObject.find(tag_i),
                                                   dim = vdim)),
                             show_edges = True,
                             edge_color = "red")
    plotter.view_xy()
    plotter.show()
    return
# }}}
# Projection problem {{{
# From: https://github.com/ericstewart36/finite_viscoelasticity/blob/main/FV01_VHB_uniaxial_tension_eq.ipynb
def setup_projection(u, V, dx):

    trial = ufl.TrialFunction(V)
    test  = ufl.TestFunction(V)

    a = ufl.inner(trial, test)*dx
    L = ufl.inner(u, test)*dx

    projection_problem = dolfinx.fem.petsc.LinearProblem(a, L, [], \
        petsc_options={"ksp_type": "cg",
                       "ksp_rtol": 1e-16,
                       "ksp_atol": 1e-16,
                       "ksp_max_it": 1000})

    return projection_problem
# }}}
# mprint {{{
# From: https://github.com/ericstewart36/finite_viscoelasticity/blob/main/FV09_NBR_bushing_shear_MPI.py
# this forces the program to still print (but only from one CPU)
# when run in parallel.
def mprint(*argv, rank = 0):
    if rank==0:
        out = ""
        for arg in argv:
            out = out + str(argv)
        print(out, flush = True)
# }}}
# L1 and L2-norm {{{
def L2norm(domain, u, dx):
    normForm = fem.form(inner(u, u)*dx)
    norm = fem.assemble_scalar(normForm)
    return np.sqrt(norm)
def L1norm(domain, u, dx):
    normForm = fem.form(sqrt(inner(u, u))*dx)
    norm = fem.assemble_scalar(normForm)
    return norm
# }}}
# From vector to matrix {{{
def FromVectorToMatrix(vector :  np.ndarray, numComp : int):
    # Check compatibility
    vSize = vector.shape[0]
    if not vSize%numComp == 0:
        raise("Incompatible vector size with number of components")
    # Initialise matrix
    numRows = int(vSize/numComp)
    matrix = np.zeros((numRows, numComp))
    # Assign values
    for k1 in range(numRows):
        for k2 in range(numComp):
            matrix[k1, k2] = vector[numComp*k1 + k2]
    return matrix
# }}}
# Plot 2D lines {{{
def Plot2DLines(df, xkey, ykeys, **kwargs):
    # Get figure size
    cm = 1.0 / 2.54
    figsize = kwargs.get("figsize", [8.0 * cm, 6.0 * cm])
    # Initialisation of figure
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    # Get x axis data
    x = df[xkey[0]] if isinstance(xkey, list) else df[xkey]
    # Plot lines
    for ykey in ykeys:
        ax.plot(x, df[ykey], label=ykey)
    # Set x label
    ax.set_xlabel(r"$t$")
    # External configuration
    figConf = kwargs.get("figConf", lambda fi, ai: (fi, ai))
    fig, ax = figConf(fig, ax)
    return fig, ax
