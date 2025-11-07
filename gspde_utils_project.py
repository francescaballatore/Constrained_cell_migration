# Libraries {{{
import dolfinx
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, io, plot, log
from dolfinx.fem.petsc import NonlinearProblem, assemble_matrix, create_matrix
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import Constant, Function, dirichletbc, Expression
from dolfinx.io import XDMFFile, VTKFile
from dolfinx.la import create_petsc_vector
from scipy.spatial import cKDTree

import ufl
from ufl import (TestFunctions, TrialFunction, Identity, grad, inner, det, div, dot, inv, tr, as_vector, outer, derivative, dev, sqrt, eq)

import basix
from basix.ufl import element, quadrature_element

import os, sys
import numpy as np
import pandas as pd
from datetime import datetime
import pyvista
import gmsh
from shapely.geometry import Polygon
from pdb import set_trace

# In-house modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from output_utils import *
from mesh_utils import *
from misc_utils import *
from filopodia_utils import *
from curvature_utils import *
# }}}

# Class GSPDE {{{
class GSPDE(object):
    # __init__ {{{
    def __init__(self, other_gspde=None, **kwargs):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.numRanks = self.comm.Get_size()
        self.kwargs = kwargs
        self.other_gspde = other_gspde
        # Set domain
        self.SetDomain()
        # Set measures
        self.SetMeasures(**kwargs)
        # Set finite element spaces
        self.SetFESpaces()
        # Set functions and constants
        self.SetVariables()
        # Set expressions
        self.SetExpressions(**kwargs)
        # Set initialisation
        self.SetInitialisation()
        # Set weak form
        self.SetWeakForm(**kwargs)
        # Set nonlinear problem
        self.SetNonlinearProblem()
        return
    # }}}
    # Set domain {{{
    def SetDomain(self):
        model = self.kwargs["model"]
        # Mesh
        self.domain, self.cellTags, self.facetTags = io.gmshio.model_to_mesh(model, self.comm, 0, gdim = 2)
        # Get dimensions
        self.dimSur = self.domain.topology.dim
        self.dimSpa = self.dimSur + 1
        # Data distribution
        self.imap = self.domain.geometry.index_map()
        self.global_node_ids = self.imap.local_to_global(np.arange(self.domain.geometry.x.shape[0]))
        self.gather_global_node_ids = self.comm.allgather(self.global_node_ids)
        local_node_ids = self.imap.global_to_local(np.arange(self.imap.size_global))
        self.node_ids_arg = np.argwhere(local_node_ids >= 0).flatten()
        self.node_ids = local_node_ids[self.node_ids_arg]
        # Get number of nodes
        self.numNods, _ = self.domain.geometry.x.shape
        # Get ordered list of node ids
        connectivities = self.domain.geometry.dofmap
        connectivities = np.array([self.imap.local_to_global(local_ele) for local_ele in connectivities])
        connectivities = self.comm.allgather(connectivities)
        connectivities = np.concatenate([array for array in connectivities if array.size > 0])
        self.numEles, nnod = connectivities.shape
        if nnod == 2:
            connectivities = np.column_stack([connectivities,
                                              -np.ones([self.numEles, 1], dtype = int)])
        self.global_orderedNodeIds = OrderNodeList(connectivities[0, 0],
                                                   connectivities[0, 0],
                                                   connectivities, self.numEles)
        local_orederedNodeIds = self.imap.global_to_local(self.global_orderedNodeIds)
        self.orderedNodeArg = np.argwhere(local_orederedNodeIds >= 0).flatten()
        self.orderedNodeIds = local_orederedNodeIds[self.orderedNodeArg]
        return
    # }}}
    # Set measures {{{
    def SetMeasures(self, **kwargs):
        self.normalDirection = kwargs.get("normalDirection", 1.0)
        quadrature_degree = self.kwargs["quadrature_degree"]
        self.x = ufl.SpatialCoordinate(self.domain)
        self.n = ufl.CellNormal(self.domain)
        self.dx = ufl.Measure("dx", domain = self.domain,
                              metadata = {"quadrature_degree" : quadrature_degree,
                                          "quadrature_rule" : "default"})
        return
    # }}}
    # Set finite element spaces {{{
    def SetFESpaces(self):
        meshOrder = self.kwargs["meshOrder"]
        # Element types: quadratic scalar
        ele_scalar = element("Lagrange", self.domain.basix_cell(), meshOrder)
        self.V_scalar = fem.functionspace(self.domain, ele_scalar)
        # Element types: quadratic vector
        ele_u = element("Lagrange", self.domain.basix_cell(), meshOrder,
                        shape = (self.dimSpa, ))
        self.V_u = fem.functionspace(self.domain, ele_u)
        # Element types: quadratic tensor
        ele_T = element("Lagrange", self.domain.basix_cell(), meshOrder, shape=(self.dimSpa, self.dimSpa))
        self.V_tensor = fem.functionspace(self.domain, ele_T)
        # Mixed element
        ele_mixed = basix.ufl.mixed_element([ele_u, ele_scalar])
        self.V_mixed = fem.functionspace(self.domain, ele_mixed)
        return
    # }}}
    # Set variables {{{
    def SetVariables(self):
        aRef = self.kwargs["aRef"]
        Dia = self.kwargs["Dia"]
        periRef = self.kwargs["periRef"]
        dt = self.kwargs["dt"]
        t = self.kwargs.get("t", 0.0)
        opre0 = self.kwargs["opre0"]
        gamma = self.kwargs.get("gamma", 0.0)
        width = self.kwargs["width"]
        length = self.kwargs["length"]
        height = self.kwargs["height"]
        x_left = self.kwargs["x_left"]
        omega = self.kwargs["omega"]
        steepness = self.kwargs["steepness"]
        strength = self.kwargs["strength"]
        Fc = self.kwargs["Fc"]
        E = self.kwargs["E"] 
        nu = self.kwargs["nu"] 
        eta1 = self.kwargs["eta1"] 
        eta2 = self.kwargs["eta2"] 
        k_n = self.kwargs["k_n"] 
        factor = self.kwargs["factor"] 
        alpha = self.kwargs["alpha"] 
        k_rep = self.kwargs["k_rep"] 
        k_t = self.kwargs["k_t"] 
        role = self.kwargs["role"]
        # Floats
        self.dt = dt
        self.aRef = aRef
        self.periRef = periRef
        self.area = aRef
        self.Dia = Dia
        self.perimeter = periRef
        self.gamma = gamma
        self.width = width
        self.length = length
        self.height = height
        self.x_left = x_left
        self.omega = omega
        self.steepness = steepness
        self.strength = strength
        self.Fc = Fc
        self.E = E
        self.nu = nu
        self.eta1 = eta1
        self.eta2 = eta2
        self.k_n = k_n
        self.factor = factor
        self.alpha = alpha
        self.k_rep = k_rep
        self.k_t = k_t
        self.role = role
        self.x_front = Dia/2
        self.x_front_p = 1
        self.x_rear = -Dia/2
        self.x_rear_p = 1
        # Constants
        self.dk = Constant(self.domain, PETSc.ScalarType(dt))
        self.t_constant = Constant(self.domain, PETSc.ScalarType(t))
        self.opre = Constant(self.domain, PETSc.ScalarType(opre0))
        # Main functions
        self.w = Function(self.V_mixed)
        self.u, self.H = ufl.split(self.w)
        self.u_test, self.H_test = TestFunctions(self.V_mixed)
        self.dw = TrialFunction(self.V_mixed)
        # Scalar functions
        self.H_old = Function(self.V_scalar)
        self.selfRepuForce = Function(self.V_scalar)
        self.bendingStiffness = Function(self.V_scalar)
        self.tensionStiffness = Function(self.V_scalar)
        self.barrierForce = Function(self.V_scalar)
        self.mechForce = Function(self.V_scalar)
        self.movForce = Function(self.V_scalar)
        self.retainForce = Function(self.V_scalar)
        self.nucleusForce = Function(self.V_scalar)
        self.repulsiveForce = Function(self.V_scalar)
        self.phi = Function(self.V_scalar)
        self.totalForce = Function(self.V_scalar)
        # Vector functions
        self.x_old = Function(self.V_u)
        self.x0 = Function(self.V_u)
        self.disp = Function(self.V_u)
        self.normal = Function(self.V_u)
        self.filoDir = Function(self.V_u)
        return
    # }}}
    # Set expressions {{{
    def SetExpressions(self, **kwargs):
        self.normal_expr = Expression(self.n, self.V_u.element.interpolation_points())
        self.x_expr = Expression(self.w.sub(0), self.V_u.element.interpolation_points())
        self.H_expr = Expression(self.w.sub(1), self.V_scalar.element.interpolation_points())
        self.disp_expr = Expression(self.x_old - self.x0, self.V_u.element.interpolation_points())
        self.totalForce_expr = Expression(self.opre
                                          + self.barrierForce
                                          + self.selfRepuForce
                                          + self.mechForce
                                          + self.movForce
                                          + self.retainForce
                                          + self.nucleusForce
                                          + self.repulsiveForce,
                                            self.V_scalar.element.interpolation_points())
        return
    # }}}
    # Set initialisation {{{
    def SetInitialisation(self):
        # Initial normal vector
        self.normal.interpolate(self.normal_expr)
        # Initial x (identity map)
        self.x_expr_id = Expression(self.x, self.V_u.element.interpolation_points())
        self.x_old.interpolate(self.x_expr_id)
        self.x0.interpolate(self.x_expr_id)
        # Initial curvature
        InitialCurvature(self.H_old, self.normal, self.x_old, self.dx)
        # Initialisation of phi
        orderedPhi = np.linspace(0.0, 2.0*np.pi, self.global_orderedNodeIds.size)
        self.phi.x.array[self.orderedNodeIds] = orderedPhi[self.orderedNodeArg]
        # Tension and bending stiffness
        self.bendingStiffness.x.array[:] = 1.0e-12
        self.tensionStiffness.x.array[:] = 1.0e-12
        # Internal variable
        self.eps_v_old = fem.Function(self.V_tensor)
        self.eps_v_old.x.array[:] = 0.0
    # }}}
    # Set weak form {{{
    def SetWeakForm(self, **kwargs):
        #Href = self.kwargs["Href"]
        Mu = (self.omega/self.dk)*inner(inner(self.u - self.x_old, self.normal), self.H_test)*self.dx
        Su = self.bendingStiffness*inner(grad(self.H), grad(self.H_test))*self.dx
        Hpow2 = self.H_old**2.0
        #Hpow2 = Href**2.0 - self.H**2.0
        Qu = -0.5*self.bendingStiffness*inner(Hpow2*self.H, self.H_test)*self.dx
        Tu = inner(self.tensionStiffness*self.H, self.H_test)*self.dx
        Fu = inner(self.totalForce, self.H_test)*self.dx
        Res_u = Mu + Su + Qu - Fu + Tu # V = -div(H) - 0.5 H^3
        MH = inner(self.H*self.normal, self.u_test)*self.dx
        SH = inner(grad(self.u), grad(self.u_test))*self.dx
        Res_H = MH - SH # H = div(x)

        self.Res = Res_u + Res_H
        self.tangent = derivative(self.Res, self.w, self.dw)
        return
    # }}}
    # Update variables {{{
    def UpdateVariables(self):
        equidistribute = self.kwargs.get("equidistribute", True)
        # Update current position and curvature
        self.x_old.interpolate(self.x_expr)
        self.H_old.interpolate(self.H_expr)
        # Update mesh
        uMat = FromVectorToMatrix(self.x_old.x.array, self.dimSpa)
        self.domain.geometry.x[:, :self.dimSpa] = uMat
        # Update area and osmotic pressure
        global_uMat = self.GetGlobalArray(uMat)
        global_orderedNodes = global_uMat[self.global_orderedNodeIds]
        xCoor = global_orderedNodes[:-1, 0]
        yCoor = global_orderedNodes[:-1, 1]
        poly = Polygon(zip(xCoor, yCoor))
        self.area = poly.area
        self.perimeter = poly.length
        # Mesh tangential movement for equidistribution
        if equidistribute:
            global_newOrderedNodes = EquidistributeMesh(global_orderedNodes, optimal = False)
            self.domain.geometry.x[self.orderedNodeIds, :self.dimSpa] = global_newOrderedNodes[self.orderedNodeArg]
            self.x_old.interpolate(self.x_expr_id)
        # Update total displacement
        self.disp.interpolate(self.disp_expr)
        # Update normal
        self.normal.interpolate(self.normal_expr)
        # Interpolate phi if tangential movement is applied
        if equidistribute:
            phi = self.phi.x.array[:]
            global_phi = self.GetGlobalArray(phi)
            global_orderedPhi = global_phi[self.global_orderedNodeIds]
            global_newOrderedPhi = CurveInterpolation(global_orderedNodes,
                                                      global_orderedPhi,
                                                      global_newOrderedNodes)
            self.phi.x.array[self.orderedNodeIds] = global_orderedPhi[self.orderedNodeArg]
        #compute x_front and x_rear
        x_array = self.x_old.x.array.reshape(-1, self.dimSpa)
        x_coords = x_array[:, 0]  # componente x
        self.x_front = np.max(x_coords)
        self.x_front_p = np.argmax(x_coords)
        self.x_rear = np.min(x_coords)
        self.x_rear_p = np.argmin(x_coords)
        return
    # }}}
    # Update loads {{{
    def UpdateLoads(self):
        # Osmotic pressure
        self.OsmoticPressure()
        # Self-repulsive force
        self.SelfRepulsiveForce()
        # Barrier force
        self.BarrierForce()
        # Movement force
        self.MovForce()
        #Retaining force
        self.RetainForce()
        # Mechanical force
        self.MechForce()
        # Nucleus to cytoplasm force
        self.NucleusToCytoplasmForce()
        # Repulsive force
        self.RepulsiveForce()
        # Update total force
        self.totalForce.interpolate(self.totalForce_expr)
        return
    # }}}
    # Osmotic pressure {{{
    def OsmoticPressure(self):
        typeOpressure = self.kwargs.get("typeOpressure", "area")
        if typeOpressure == "area":
            UpdateOpressure(self.opre, self.area, self.aRef, self.dt)
        elif typeOpressure == "perimeter":
            UpdateOpressure(self.opre, self.perimeter, self.periRef, self.dt)
        # elif typeOpressure == "both":
        #     periFactor = self.kwargs.get("periFactor", 2.0)
        #     UpdateOpressure_area_perimeter(self.opre, self.area, self.aRef,
        #                                    self.perimeter, self.periRef*periFactor,
        #                                    self.dt, self.alpha, self.beta, self.gamma)
        elif typeOpressure == "none":
            self.opre.value = 0.0
        else:
            message = "Invalid type of osmotic pressure. Use: 'area', 'perimeter' or 'none'"
            raise TypeError(message)
        return
    
    # Self-repulsive force {{{
    def SelfRepulsiveForce(self):
        rep_normal_tol = self.kwargs.get("rep_normal_tol", -0.7)
        rep_tol = self.kwargs["rep_tol"]
        rep_mag = self.kwargs["rep_mag"]
        rep_st  = self.kwargs["rep_st"]
        normalArray = FromVectorToMatrix(self.normal.x.array, self.dimSpa)
        global_normalArray = self.GetGlobalArray(normalArray)
        xArray = self.domain.geometry.x[:, :self.dimSpa]
        global_xArray = self.GetGlobalArray(xArray)
        repuForce = SelfRepulsiveForce_kdtree(xArray, normalArray,
                                              global_xArray, global_normalArray,
                                              rep_normal_tol, rep_st, rep_tol,
                                              rep_mag, rep_st)
        self.selfRepuForce.x.array[:] = repuForce
        return
    # }}}

    # Mechanical force
    def MechForce(self):
        u_diff = self.x - self.x0

        typeModel = self.kwargs.get("mech_model", "elasticity")
        if typeModel == "elasticity":
            mu = self.E / (2 * (1 + self.nu))
            lmbda = self.E * self.nu / ((1 + self.nu)*(1 - 2 * self.nu))

            def epsilon(u):
                return ufl.sym(ufl.grad(u))

            def sigma(u):
                return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(self.dimSpa) + 2 * mu * epsilon(u)

            # Proiezione della forza elastica lungo la normale
            elastic_force_vector = ufl.dot(sigma(u_diff), self.normal)
            scalar_normal_component = ufl.dot(elastic_force_vector, self.normal)
            self.avg_sigma_n = fem.assemble_scalar(fem.form(scalar_normal_component * self.dx)) / fem.assemble_scalar(fem.form(1.0 * self.dx))

            V_m = self.mechForce.function_space
            q_m = ufl.TestFunction(V_m)
            p_m = ufl.TrialFunction(V_m)

            a_m = ufl.inner(p_m, q_m) * self.dx
            L_m = ufl.inner(scalar_normal_component, q_m) * self.dx

            problem_m = fem.petsc.LinearProblem(a_m, L_m, bcs=[], u=self.mechForce,
                                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
            problem_m.solve()

        elif typeModel == "viscoelasticity":
            tau = (self.eta1 + self.eta2) / self.E
            tau2 = self.eta2 / self.E

            def epsilon(u):
                 return ufl.sym(ufl.grad(u))

            # formula UFL per il nuovo tensore viscoelastico
            eps_v_new = (self.eps_v_old + (self.dt / tau2) * epsilon(u_diff)) / (1.0 + self.dt / tau)

            # tensore di Cauchy: sigma = E (ε - ε^v)
            stress_tensor = self.E * (epsilon(u_diff) - eps_v_new)
            viscoelastic_force_vector = ufl.dot(stress_tensor, self.normal)
            scalar_normal_component = ufl.dot(viscoelastic_force_vector, self.normal)
            self.avg_sigma_n = fem.assemble_scalar(fem.form(scalar_normal_component * self.dx)) \
                            / fem.assemble_scalar(fem.form(1.0 * self.dx))

            # risolvo il problema per la forza
            V_m = self.mechForce.function_space
            q_m = ufl.TestFunction(V_m)
            p_m = ufl.TrialFunction(V_m)

            a_m = ufl.inner(p_m, q_m) * self.dx
            L_m = ufl.inner(scalar_normal_component, q_m) * self.dx

            problem_m = fem.petsc.LinearProblem(a_m, L_m, bcs=[], u=self.mechForce,
                                                petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
            problem_m.solve()

            expr = fem.Expression(eps_v_new, self.V_tensor.element.interpolation_points())
            self.eps_v_old.interpolate(expr)

        return

    # }}}

    ### External forces
    # Barrier force (both cell and nucleus)
    def BarrierForce(self):
        y_top_inner = +self.width / 2
        y_bot_inner = -self.width / 2

        # Forza totale
        total_vector_barrier_force = ufl.as_vector([0.0] * self.dimSpa)
        
        # Versori diretti lungo y
        y_unit_up = ufl.as_vector([0.0, 1.0])
        y_unit_down = ufl.as_vector([0.0, -1.0])

        in_x_range = ufl.And(ufl.ge(self.x[0], self.x_left), ufl.le(self.x[0], self.x_left + self.length))

        # Penetrazione nella parete superiore: distanza da y_top_inner
        penetration_top = self.x[1] - y_top_inner

        # Penetrazione nella parete inferiore: distanza da y_bot_inner
        penetration_bot = y_bot_inner - self.x[1]

        # Forza nella parete superiore (verso il basso, se penetra e dentro il rettangolo)
        force_mag_top = ufl.conditional(
            in_x_range,
            self.strength * ufl.exp(self.steepness * penetration_top),
            0.0
        )
        total_vector_barrier_force += force_mag_top * y_unit_down

        # Forza nella parete inferiore (verso l’alto, se penetra e dentro il rettangolo)
        force_mag_bot = ufl.conditional(
            in_x_range,
            self.strength * ufl.exp(self.steepness * penetration_bot),
            0.0
        )
        total_vector_barrier_force += force_mag_bot * y_unit_up

        # Centri dei semicerchi: (sinistri e destri)
        centres = [
            ufl.as_vector([self.x_left, +self.width/2 + self.height/2]),  # top-left
            ufl.as_vector([self.x_left, -self.width/2 - self.height/2]),  # bot-left
            ufl.as_vector([self.x_left+self.length, +self.width/2 + self.height/2]),  # top-right
            ufl.as_vector([self.x_left+self.length, -self.width/2 - self.height/2])   # bot-right
        ]

        in_x_range = ufl.Or(ufl.le(self.x[0], self.x_left), ufl.ge(self.x[0], self.x_left + self.length))

        for c_vec in centres:
            delta = self.x - c_vec
            dist = sqrt(dot(delta, delta) + 1e-6)
            f_dir = delta / dist
            penetration = self.height/2 - dist
            force_mag = ufl.conditional(
                in_x_range,
                self.strength * ufl.exp(self.steepness * penetration),
                0.0)
            total_vector_barrier_force += force_mag * f_dir

        # Proiezione sulla normale
        scalar_normal_component = ufl.dot(total_vector_barrier_force, self.normal)

        # Risoluzione del problema FEM
        V_b = self.barrierForce.function_space
        q_b = ufl.TestFunction(V_b)
        p_b = ufl.TrialFunction(V_b)
        a_b = inner(p_b, q_b) * self.dx
        L_b = inner(scalar_normal_component, q_b) * self.dx

        problem_b = fem.petsc.LinearProblem(a_b, L_b, bcs=[], u=self.barrierForce,
                                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        problem_b.solve()

        return
    
    # def BarrierForce(self):
    #     total_vector_barrier_force = ufl.as_vector([0.0] * self.dimSpa)
        
    #     for c_vec in self.centres_list: 

    #         delta = self.x - c_vec
    #         x_mag = sqrt(dot(delta, delta) + 1.0e-6) 
            
    #         f_dir = delta / x_mag # Direction from barrier center to membrane point
            
    #         penetration = self.r_barrier - x_mag
            
    #         force_mag = ufl.conditional(
    #             ufl.gt(penetration, 0.0),
    #             self.strength * ufl.exp(self.steepness * penetration),
    #             0.0
    #          )
    #         total_vector_barrier_force += force_mag * f_dir 
            
    #     scalar_normal_component = ufl.dot(total_vector_barrier_force, self.normal)

    #     V_b = self.barrierForce.function_space 
    #     q_b = ufl.TestFunction(V_b)
    #     p_b = ufl.TrialFunction(V_b) 

    #     a_b = inner(p_b, q_b) * self.dx
    #     L_b = inner(scalar_normal_component, q_b) * self.dx

    #     problem_b = fem.petsc.LinearProblem(a_b, L_b, bcs=[], u=self.barrierForce, 
    #                                          petsc_options={"ksp_type": "preonly", "pc_type": "lu"}) 
    #     problem_b.solve()

    #     return 
    # }}}

    # Pressure to push cell (cell only)
    def MovForce(self):
        if self.dimSpa == 2:
            f_dir_vec = [1.0, 0.0]
        else:  # self.dimSpa == 3
            f_dir_vec = [1.0, 0.0, 0.0]

        f_dir = ufl.as_vector(f_dir_vec)
        # x_c = compute_center(self.domain)
        # centre_dir = ufl.as_vector(self.x - x_c)
        # norm_centre_dir = ufl.sqrt(ufl.dot(centre_dir, centre_dir))
        # centre_dir /= norm_centre_dir
        f_cyto = self.Fc * f_dir
        scalar_normal_component = ufl.dot(f_cyto, self.normal)

        V_m = self.movForce.function_space
        q_m = ufl.TestFunction(V_m)
        p_m = ufl.TrialFunction(V_m)

        a_m = inner(p_m, q_m) * self.dx
        L_m = inner(scalar_normal_component, q_m) * self.dx

        problem_m = fem.petsc.LinearProblem(a_m, L_m, bcs=[], u=self.movForce,
                                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        problem_m.solve()
        return
    
        # Force that avoid the contact and the intersection between cell and nucleus (cell and nucleus)
    def RepulsiveForce(self):
        if self.role == "nucleus":
            V_c = self.repulsiveForce.function_space
            delta = compute_distance_to_cortex(self.domain, self.other_gspde.domain, V_c)
            phi_delta = ufl.exp(-self.alpha * delta)
            f_rep = -self.k_rep * phi_delta 
            #f_rep = 1e-10
        elif self.role == "cell":
            V_c = self.repulsiveForce.function_space
            delta = compute_distance_to_cortex(self.domain, self.other_gspde.domain, V_c)
            phi_delta = ufl.exp(-self.alpha * delta)
            f_rep = self.k_rep * phi_delta 
            #f_rep = 1e-10
        else:
            f_rep = 1e-10 #ufl.zero()

        # Variational form
        V_c = self.repulsiveForce.function_space
        q_c = ufl.TestFunction(V_c)
        p_c = ufl.TrialFunction(V_c)

        a_c = inner(p_c, q_c) * self.dx
        L_c = inner(f_rep, q_c) * self.dx

        # Solve and store in self.repulsiveForce
        problem = fem.petsc.LinearProblem(
            a_c, L_c, u=self.repulsiveForce, bcs=[],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        problem.solve()
    
    # Force that slows the cell when the nucleus is far (cell only)
    def RetainForce(self):
         V_r = self.retainForce.function_space
         if self.role == "cell":
            dist = compute_distance_to_cortex(self.domain, self.other_gspde.domain, V_r)
            dist_ref = self.Dia/2 - self.other_gspde.Dia/2
            dist_weight = ufl.conditional(ufl.gt(dist, dist_ref),dist - dist_ref,1e-8)
            f_int = -self.k_t*dist_weight
         else:
             # Nessuna forza se la cellula non è nota
             f_int = ufl.zero()

         q_r = ufl.TestFunction(V_r)
         p_r = ufl.TrialFunction(V_r)

         a_r = inner(p_r, q_r) * self.dx
         L_r = inner(1e-6 + f_int, q_r) * self.dx  

         problem = fem.petsc.LinearProblem(a_r, L_r, bcs=[], u=self.retainForce,
                                         petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
         problem.solve()
         return
    
    # def RetainForce(self):
    #     V_r = self.retainForce.function_space
    #     if self.role == "cell":
    #         max_H = self.other_gspde.H_old.x.array.max()
    #         min_H = self.other_gspde.H_old.x.array.min()
    
    #         dist = compute_distance_to_cortex(self.domain, self.other_gspde.domain, V_r)
    #         dist_ref = self.Dia/2 - self.other_gspde.Dia/2
    #         dist_weight = ufl.conditional(ufl.gt(dist, dist_ref),dist - dist_ref,1e-8)

    #         if np.isclose(max_H, 1/(self.other_gspde.Dia/2), atol=1e-2): 
    #             coeff = self.k_t
    #             self.counter = 0
    #         elif np.isclose(max_H, 2/(self.other_gspde.Dia/2), atol=5e-2) and min_H > -5e-2:
    #             self.counter = max(self.counter-1,0)
    #             factor = max(self.counter / 10, 0.0)
    #             coeff = self.k_t * (1.0 + factor)
    #         else:
    #             self.counter = min(self.counter+1,10)
    #             factor = min(self.counter / 10, 1.0)
    #             coeff = self.k_t * (1 + factor)

    #         print("coeff:", coeff)
    #         f_int = -coeff*dist_weight
    #     else:
    #         f_int = ufl.zero()

    #     q_r = ufl.TestFunction(V_r)
    #     p_r = ufl.TrialFunction(V_r)

    #     a_r = inner(p_r, q_r) * self.dx
    #     L_r = inner(1e-6 + f_int, q_r) * self.dx  

    #     problem = fem.petsc.LinearProblem(a_r, L_r, bcs=[], u=self.retainForce,
    #                                     petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    #     problem.solve()
    #     return

    # Force that link nucleus to cell (nucleus only)
    def NucleusToCytoplasmForce(self):
        if self.role == "nucleus":
            # Calcola i centroidi del nucleo e della cellula
            #x_n = compute_center(self.domain)
            #x_c = compute_center(self.other_gspde.domain)
            #delta = x_c - x_n
            
            #delta = ufl.as_vector(delta[:2].tolist())
            #norm_delta = ufl.sqrt(ufl.dot(delta, delta)) + 1e-8  # evita divisione per zero
            #delta_normalized = (delta / norm_delta)
            delta_normalized = ufl.as_vector([1.0, 0.0])
            projection = ufl.dot(delta_normalized, self.normal) 

            H_at_front = self.H_old.x.array[self.x_front_p]
            H_at_rear = self.H_old.x.array[self.x_rear_p]

            if np.isclose(H_at_front, H_at_rear, atol=1e-2): 
                 coeff = self.k_n
            else:
                 coeff = self.k_n/self.factor

            f_int = coeff * projection
        else:
            f_int = ufl.zero()

        V_n = self.nucleusForce.function_space
        q_n = ufl.TestFunction(V_n)
        p_n = ufl.TrialFunction(V_n)

        a_n = inner(p_n, q_n) * self.dx
        L_n = inner(1e-8 + f_int, q_n) * self.dx  # (aggiunta costante per evitare zero identico)

        problem = fem.petsc.LinearProblem(a_n, L_n, bcs=[], u=self.nucleusForce,
                                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        problem.solve()
        return

    # Set nonlinear problem {{{
    def SetNonlinearProblem(self):
        SetSolverOpt = self.kwargs["SetSolverOpt"]
        self.problem = NonlinearProblem(self.Res, self.w, [], self.tangent)
        self.solver = NewtonSolver(self.comm, self.problem)
        SetSolverOpt(self.solver)
        return
    # }}}
    # Solve {{{
    def Solve(self):
        iters, converged = self.solver.solve(self.w)
        return iters, converged
    # }}}
    # Get global array {{{
    def GetGlobalArray(self, array):
        # Gather array
        gather_array = self.comm.allgather(array)
        # Initialisation of global array
        if len(array.shape) == 1:
            size = [self.imap.size_global]
        elif len(array.shape) == 2:
            _, cols = array.shape
            size = [self.imap.size_global, cols]
        else:
            raise("Not yet available for arrays of len(shape) > 2")
        global_array = np.zeros(size)
        # Fill array
        for k1 in range(self.numRanks):
            global_array[self.gather_global_node_ids[k1]] = gather_array[k1]
        return global_array
    # }}}
# }}}

# Solve iteration {{{
def SolveIteration(ite, t, gspdes, toSolve_list):
    # Update time
    for gspde_i in gspdes:
        gspde_i.t_constant.value = t
    # Update loads
    for gspde_i in gspdes:
        gspde_i.UpdateLoads()
    # Solve problem
    for k1 in toSolve_list:
        gspde_i = gspdes[k1]
        gspde_i.Solve()
        # Collect results form MPI ghost processes
        gspde_i.w.x.scatter_forward()
    # Update variables
    for gspde_i in gspdes:
        gspde_i.UpdateVariables()
    return
# }}}

def compute_center(mesh: dolfinx.mesh.Mesh) -> np.ndarray:
    """
    Compute the geometric center (centroid) of a given mesh.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The mesh whose centroid is to be computed.

    Returns
    -------
    np.ndarray
        A 2D vector representing the centroid (x,y) of the mesh.
    """
    coords = mesh.geometry.x
    center3d = np.mean(coords, axis=0)
    center2d = center3d[:2]  # prendi solo x e y
    return center2d

def compute_distance_to_cortex(nucleus_mesh: mesh.Mesh, cortex_mesh: mesh.Mesh, V: fem.FunctionSpace) -> fem.Function:
    """
    For each point on the nucleus mesh, compute the minimal Euclidean distance 
    to the cortex mesh and return it as a fem.Function.

    Parameters
    ----------
    nucleus_mesh : dolfinx.mesh.Mesh
        The mesh of the nucleus (Γ_n).
    cortex_mesh : dolfinx.mesh.Mesh
        The mesh of the cell cortex.
    V : fem.FunctionSpace
        A scalar FunctionSpace on the nucleus mesh (e.g., CG1 or DG0).

    Returns
    -------
    delta_func : fem.Function
        A function assigning to each dof on the nucleus the minimal distance to the cortex.
    """
    # Coordinates of points on each surface
    x_n = nucleus_mesh.geometry.x  # coordinates of the nucleus mesh
    x_c = cortex_mesh.geometry.x   # coordinates of the cell (cortex)

    # Build a KDTree on the cortex points
    cortex_tree = cKDTree(x_c)

    # Query minimal distance for each point of the nucleus
    distances, _ = cortex_tree.query(x_n)

    # Create a Function and interpolate distances
    delta_func = fem.Function(V)
    delta_func.x.array[:] = distances
    delta_func.x.scatter_forward()

    return delta_func

# Adaptive time solution {{{
def AdaptiveTimeSolver(ite, tf, dt, maxForceDiff, gspdes, 
                       toSolve_list, barrierForceId_list, minStepFrac = 4.0):
    # Solve the system in a test
    run = True
    test_dt = dt
    t0 = tf - dt
    test_t = t0 + test_dt
    gspde_solve = [gspdes[k1] for k1 in toSolve_list]
    gspde_tests = [gspdes[k1] for k1 in barrierForceId_list]
    numTests = len(gspde_tests)
    while run:
        # Create a copy of the problem
        w_copy = []
        for gspde_i in gspdes:
            w_copy.append(np.copy(gspde_i.w.x.array))
        # Update time
        for gspde_i in gspdes:
            gspde_i.dk.value = test_dt
            gspde_i.t_constant.value = test_t
        # Update variables
        if ite > 1:
            for gspde_i in gspde_solve:
                gspde_i.UpdateVariables()
        # Update loads
        for gspde_i in gspdes:
            gspde_i.UpdateLoads()
        # Evaluate current force
        global_currentForces = []
        for gspde_test in gspde_tests:
            currentForce = gspde_test.barrierForce.x.array + gspde_test.selfRepuForce.x.array + gspde_test.movForce.x.array + gspde_test.mechForce.x.array + gspde_test.retainForce.x.array + gspde_test.nucleusForce.x.array + gspde_test.repulsiveForce.x.array
            global_currentForces.append(np.copy(gspde_test.GetGlobalArray(currentForce)))
        global_currentForce = np.hstack(global_currentForces)
        # Solve problems
        for gspde_i in gspde_solve:
            gspde_i.Solve()
            # Collect results form MPI ghost processes
            gspde_i.w.x.scatter_forward()
        # New ws
        new_w = []
        for gspde_i in gspdes:
            new_w.append(np.copy(gspde_i.w.x.array))
        # Check new force {{{
        for gspde_i in gspde_solve:
            gspde_i.UpdateVariables()
        # Update loads
        for gspde_i in gspdes:
            gspde_i.UpdateLoads()
        # Compute future force
        global_futureForces = []
        for gspde_test in gspde_tests:
            futureForce = gspde_test.barrierForce.x.array + gspde_test.selfRepuForce.x.array + gspde_test.movForce.x.array + gspde_test.mechForce.x.array + gspde_test.retainForce.x.array + gspde_test.nucleusForce.x.array + gspde_test.repulsiveForce.x.array 
            global_futureForces.append(np.copy(gspde_test.GetGlobalArray(futureForce)))
        global_futureForce = np.hstack(global_futureForces)
        diffForce = np.linalg.norm(np.abs(global_futureForce - global_currentForce),
                                   np.inf)
        print("--------------------")
        print("test_t ", test_t, "test_dt", test_dt)
        print("Max force difference: ", maxForceDiff, "Force difference: ", diffForce)
        # }}}
        # Check
        if ite == 1:
            run = False
        else:
            if diffForce < maxForceDiff:
                if np.isclose(test_t, tf):
                    run = False
                else:
                    test_t += test_dt
            else:
                # Check size of dt
                if np.isclose(test_dt, dt/minStepFrac):
                    # Finish while if tf is reached
                    if np.isclose(test_t, tf):
                        run = False
                    # Continue with the same dt
                    test_t += test_dt
                    print("Warning: not able to reduce the time step!")
                else:
                    # Reduce dt
                    test_dt = test_dt/2.0
                    test_t -= test_dt
                    # Go to previous solution
                    for gspde_i, w_i in zip(gspdes, w_copy):
                        gspde_i.w.x.array[:] = w_i[:]
    return new_w
# }}}