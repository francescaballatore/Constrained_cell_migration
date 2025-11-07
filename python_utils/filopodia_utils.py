# Libraries {{{
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx.fem import Constant
from petsc4py import PETSc
from pdb import set_trace
from scipy.spatial import KDTree
from shapely.geometry import Polygon, Point
# }}}

# General ea_class {{{
class ea_class(object):
    # Time function {{{
    def eaTimeFunc(self, t):
        # one and zero values
        zero = Constant(t.ufl_domain(), PETSc.ScalarType(0.0))
        one  = Constant(t.ufl_domain(), PETSc.ScalarType(1.0))
        # Compute value
        if t.value >= self.t0 and t.value < self.t0 + self.period:
            return one
        else:
            return zero
    # }}}
    # __call__ {{{
    def __call__(self, phi, t):
        eaSpace = self.eaSpaceFunc(phi)
        eaTime = self.eaTimeFunc(t)
        return eaSpace*eaTime
    # }}}
# }}}
# Class to define ea Gaussian {{{
class ea_Gaussian(ea_class):
    # Properties {{{
    @property
    def height(self):
        return self._height
    @property
    def width(self):
        return self._width
    @property
    def period(self):
        return self._period
    @property
    def theta(self):
        return self._theta
    @property
    def t0(self):
        return self._t0
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        self._theta = kwargs.get("theta", 0.0)
        self._t0 = kwargs.get("t0", 0.0)
        self._width = kwargs["width"]
        self._height = kwargs["height"]
        self._period = kwargs["period"]
        return
    # }}}
    # Space and time functions {{{
    def eaSpaceFunc(self, phi):
        return self.height*ufl.exp(-(phi - self.theta)**2.0/(2.0*self.width**2.0))
    def eaTimeFunc(self, t):
        ome = 2.0*np.pi/self.period
        return (1.0 + ufl.sin(ome*(t - self.t0) - np.pi/2.0))/2.0
    # }}}
# }}}
# Class to define ea from cosine {{{
class ea_Cos(ea_Gaussian):
    # Space and time functions {{{
    def eaSpaceFunc(self, phi):
        return self.height*ufl.cos((phi - self.theta)/2.0)**(int(self.width))
    def eaTimeFunc(self, t):
        # one and zero values
        zero = Constant(t.ufl_domain(), PETSc.ScalarType(0.0))
        one  = Constant(t.ufl_domain(), PETSc.ScalarType(1.0))
        # Compute value
        ome = 2.0*np.pi/self.period
        sinval = np.sin(ome*(t.value - self.t0))
        if sinval > 0.0:
            return one
        else:
            return zero
    # }}}
# }}}
# Class to sum up eas defined for expressions {{{
class eaSum(object):
    # Properties {{{
    @property
    def eas(self):
        return self._eas
    @property
    def N(self):
        return self._N
    @property
    def ForceClass(self):
        return self._ForceClass
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        self._N = kwargs["N"]
        hs = kwargs["hs"]
        Ts = kwargs["Ts"]
        t0s = kwargs["t0s"]
        thetas = kwargs["thetas"]
        ws = kwargs["ws"]
        ForceClass = kwargs["ForceClass"]
        # Create random parameters
        t0s -= t0s.min()
        # Create eas
        eas = [None]*self.N
        for k1 in range(self.N):
            eas[k1] = ForceClass(theta = thetas[k1],
                                  t0 = t0s[k1],
                                  width = ws[k1],
                                  period = Ts[k1],
                                  height = hs[k1])
        self._eas = eas
        return
    # }}}
    # __call__ {{{
    def __call__(self, phi, t, dt, domain):
        sum_eas = 0.0
        for ea_k1 in self.eas:
            sum_eas += ea_k1(phi, t)
        return sum_eas
    # }}}
# }}}
# Class to get maximum eas defined for expressions {{{
class eaMax(object):
    # Properties {{{
    @property
    def eas(self):
        return self._eas
    @property
    def N(self):
        return self._N
    @property
    def ForceClass(self):
        return self._ForceClass
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        self._N = kwargs["N"]
        hs = kwargs["hs"]
        Ts = kwargs["Ts"]
        t0s = kwargs["t0s"]
        thetas = kwargs["thetas"]
        ws = kwargs["ws"]
        ForceClass = kwargs["ForceClass"]
        # Create random parameters
        t0s -= t0s.min()
        # Create eas
        eas = [None]*self.N
        for k1 in range(self.N):
            eas[k1] = ForceClass(theta = thetas[k1],
                                  t0 = t0s[k1],
                                  width = ws[k1],
                                  period = Ts[k1],
                                  height = hs[k1])
        self._eas = eas
        return
    # }}}
    # __call__ {{{
    def __call__(self, phi, t, dt, domain):
        max_eas = 0.0
        for ea_k1 in self.eas:
            max_eas = ufl.max_value(max_eas, ea_k1(phi, t))
        return max_eas
    # }}}
# }}}
# Class to define ea by distance to a point {{{
class ea_Distance(ea_class):
    # Properties {{{
    @property
    def magnitude(self):
        return self._magnitude
    @property
    def width(self):
        return self._width
    @property
    def source(self):
        return self._source
    @property
    def t0(self):
        return self._t0
    @property
    def period(self):
        return self._period
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        self._magnitude = kwargs["magnitude"]
        self._width = kwargs["width"]
        self._source = kwargs["source"]
        self._t0 = kwargs["t0"]
        self._period = kwargs["period"]
        return
    # }}}
    # Space functions {{{
    def eaSpaceFunc(self, xArray, h):
        # Create KDTree
        tree = KDTree(xArray)
        # Define the number of points and effective force
        eqPressure = self.magnitude/self.width
        numForcePoints = np.ceil(self.width/h)
        dumpPressure = self.magnitude/(h*numForcePoints)
        effForce = eqPressure/dumpPressure*self.magnitude
        # Find nearest points
        distance, indices = tree.query(self.source, k = numForcePoints)
        # Find direction
        direction = self.source - xArray[indices]
        if len(direction.shape) == 1:
            direction = np.array([(direction)])
        dirMag = np.linalg.norm(direction, axis = 1)
        direction[:, 0] = direction[:, 0]/dirMag
        direction[:, 1] = direction[:, 1]/dirMag
        return indices, effForce, direction
    # }}}
    # Time function {{{
    def eaTimeFunc(self, t):
        # Compute value
        if t.value >= self.t0 and t.value < self.t0 + self.period:
            return 1.0
        else:
            return 0.0
    # }}}
    # __call__ {{{
    def __call__(self, phi, t, **kwargs):
        h = kwargs["h"]
        indices, effForce, direction = self.eaSpaceFunc(phi, h)
        eaTime = self.eaTimeFunc(t)
        return indices, effForce*eaTime, direction
    # }}}
# }}}
# Class to sum eas defined by nodes {{{
class eaSumNodal(object):
    # Properties {{{
    @property
    def eas(self):
        return self._eas
    @property
    def N(self):
        return self._N
    @property
    def ForceClass(self):
        return self._ForceClass
    # }}}
    # __init__ {{{
    def __init__(self, **kwargs):
        self._N = kwargs["N"]
        hs = kwargs["hs"]
        Ts = kwargs["Ts"]
        t0s = kwargs["t0s"]
        thetas = kwargs["thetas"]
        ws = kwargs["ws"]
        ForceClass = kwargs["ForceClass"]
        sourceDia = kwargs.get("sourceDia", 1.0e6)
        # Create random parameters
        t0s -= t0s.min()
        # Create eas
        eas = [None]*self.N
        for k1 in range(self.N):
            theta = thetas[k1]
            source = np.array([np.cos(theta), np.sin(theta)])*sourceDia
            eas[k1] = ForceClass(source = source,
                                  t0 = t0s[k1],
                                  width = ws[k1],
                                  period = Ts[k1],
                                  magnitude = hs[k1])
        self._eas = eas
        return
    # }}}
    # __call__ {{{
    def __call__(self, phi, t, **kwargs):
        indices = []
        forces = []
        directions = []
        for ea_k1 in self.eas:
            indices_k1, force_k1, direction_k1 = ea_k1(phi, t, **kwargs)
            indices.append(indices_k1)
            forces.append(np.ones(indices_k1.size)*force_k1)
            directions.append(direction_k1)
        return indices, forces, directions
    # }}}

# }}}

# Compute self-repulsive force {{{
def SelfRepulsiveForce(xArray, normalArray, normal_tol, normal_st,
                       dis_tol, dis_mag, dis_st):
    # Compute the contribution of normal {{{
    # Get normal vector matrix
    nxArray = normalArray[:, 0]
    nyArray = normalArray[:, 1]
    # Compute ni\cdot nj
    normalDotNormal = np.einsum('i,j->ij', nxArray, nxArray) + np.einsum('i,j->ij', nyArray, nyArray)
    # Filter normal dots to keep those close to -1
    normal_contribution = (np.tanh(-normal_st*(normalDotNormal - normal_tol)) + 1.0)/2.0
    # }}}
    # Compute the contribution of the distance {{{
    # Compute the distance between points
    diffArray = xArray[:, np.newaxis, :] - xArray[np.newaxis, :, :]
    Dij_squared = np.einsum('ijk,ijk->ij', diffArray, diffArray)
    Dij = np.sqrt(Dij_squared)
    # Accentuate distance contribution
    dis_contribution = dis_mag*np.exp(-dis_st*(Dij - dis_tol))
    # }}}
    # Repulsive force
    fullContribution = normal_contribution*dis_contribution
    repuForce = np.max(fullContribution, axis = 1)
    return repuForce
def SelfRepulsiveForce_parallel(local_xArray, local_normalArray,
                                global_xArray, global_normalArray,
                                normal_tol, normal_st, dis_tol, dis_mag, dis_st):
    comm = MPI.COMM_WORLD
    from datetime import datetime
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_numNods, _ = local_xArray.shape
    global_numNods, _ = global_xArray.shape
    # Compute the contribution of normal {{{
    # Compute ni\cdot nj
    normalDotNormal = np.einsum('i,j->ij', local_normalArray[:, 0],
                                global_normalArray[:, 0]) \
                    + np.einsum('i,j->ij', local_normalArray[:, 1],
                                global_normalArray[:, 1])
    # Filter normal dots to keep those close to -1
    normal_contriution = (np.tanh(-normal_st*(normalDotNormal - normal_tol)) + 1.0)/2.0
    # }}}
    # Compute the contribution of the distance {{{
    # Compute the distance between points
    diffArray = local_xArray[:, np.newaxis, :] - global_xArray[np.newaxis, :, :]
    Dij_squared = np.einsum('ijk,ijk->ij', diffArray, diffArray)
    Dij = np.sqrt(Dij_squared)
    # Accentuate distance contribution
    dis_contribution = dis_mag*np.exp(-dis_st*(Dij - dis_tol))
    # }}}
    # Repulsive force
    fullContribution = normal_contriution*dis_contribution
    repuForce = np.max(fullContribution, axis = 1)
    return repuForce

def SelfRepulsiveForce_kdtree(local_xArray, local_normalArray,
                              global_xArray, global_normalArray,
                              normal_tol, normal_st, dis_tol, dis_mag,
                              dis_st, dis_tol_factor = 10.0):
    local_numPoints, _ = local_xArray.shape
    # Set up KDTree
    tree = KDTree(global_xArray)
    # Find indices of closest points
    closest_point_ids = tree.query_ball_point(local_xArray, dis_tol*dis_tol_factor)
    # Compute repulsive force from distance and dot normals
    repuForce = np.zeros(local_numPoints)
    for k1 in range(local_numPoints):
        # Get local point and normal and ids of closest points
        point = local_xArray[k1]
        normal = local_normalArray[k1]
        ids = closest_point_ids[k1]
        # Compute distance from local point to closest points
        distance_point = np.linalg.norm(global_xArray[ids] - point,
                                        axis = 1)
        distance_contribution = dis_mag*np.exp(-dis_st*(distance_point - dis_tol))
        # Compute normal contribution
        global_normals = global_normalArray[ids]
        dots = np.dot(global_normals, normal)
        normal_contribution = (np.tanh(-normal_st*(dots - normal_tol)) + 1.0)/2.0
        # Compute force
        full_contribution = normal_contribution*distance_contribution
        repuForce[k1] = np.max(full_contribution)
    return repuForce
#}}}
def SelfRepulsiveForce_full(local_xArray, local_normalArray,
                            global_xArray, global_normalArray,
                            normal_tol, normal_st, dis_tol, dis_mag,
                            dis_st):
    local_numPoints = local_xArray.shape[0]
    global_numPoints = global_xArray.shape[0]
    repuForce = np.zeros(local_numPoints)

    for k1 in range(local_numPoints):
        point = local_xArray[k1]
        normal = local_normalArray[k1]

        # Compute distance to all global points
        distance_point = np.linalg.norm(global_xArray - point, axis=1)
        distance_contribution = dis_mag * np.exp(-dis_st * (distance_point - dis_tol))

        # Compute normal contribution
        dots = np.dot(global_normalArray, normal)
        normal_contribution = (np.tanh(-normal_st * (dots - normal_tol)) + 1.0) / 2.0

        # Combine contributions
        full_contribution = normal_contribution * distance_contribution

        # Take max contribution (or sum/mean depending on model)
        repuForce[k1] = np.max(full_contribution)

    return repuForce

# Osmotic pressure {{{
def UpdateOpressure(opre, A, Aref, dt):
    oldPressure = opre.value
    Kp = 120000.0
    Ki = 500.0
    newPressure = (oldPressure + dt*Kp*(Aref - A)/Aref)/(1.0 + Ki*dt)
    opre.value = newPressure
    return
# }}}
# Cell-basement membrane contact force {{{
def CellBmContactForce(xCell, xBm, dis_mag, dis_st, dis_tol,
                       inOrder = False):
    # Expand arrays
    xCell_expanded = xCell[:, np.newaxis, :]
    xBm_expanded = xBm[np.newaxis, :, :]
    # Compute difference vectors
    diff = xCell_expanded - xBm_expanded
    # Compute distance array
    Dij_squared = np.einsum('ijk,ijk->ij', diff, diff)
    Dij = np.sqrt(Dij_squared)
    # Give sign to distance if the arrays are in order
    if inOrder:
        # Define polygons
        xCoor_ce = xCell[:, 0]
        yCoor_ce = xCell[:, 1]
        poly_ce = Polygon(zip(xCoor_ce, yCoor_ce))
        xCoor_bm = xBm[:, 0]
        yCoor_bm = xBm[:, 1]
        poly_bm = Polygon(zip(xCoor_bm, yCoor_bm))
        # Search for points in cell outside the basement membrane zone
        cell_out_bm = np.array([not poly_bm.contains(Point(pi)) for pi in xCell])
        cell_out_bm_args = np.argwhere(cell_out_bm == True)
        bm_in_cell = np.array([poly_ce.contains(Point(pi)) for pi in xBm])
        bm_in_cell_args = np.argwhere(bm_in_cell == True)
        # Make negative distances at penetrating pairs
        if bm_in_cell_args.size*cell_out_bm_args.size > 0:
            bm_in_cell_ids = np.hstack(bm_in_cell_args)
            cell_out_bm_ids = np.hstack(cell_out_bm_args)
            for k1 in cell_out_bm_ids:
                for k2 in bm_in_cell_ids:
                    Dij[k1, k2] = - Dij[k1, k2]
    # Compute repelling force
    repForce = dis_mag*np.exp(-dis_st*(Dij - dis_tol))
    # Set repelling force as the maximum at each node
    repForce_ce = np.max(repForce, axis = 1)
    repForce_bm = np.max(repForce, axis = 0)
    return repForce_ce, repForce_bm
# }}}
# Contact force towards obstacle {{{
def ContactForceObstacle(xArray, xObst, dis_mag, dis_st, dis_tol,
                         inOrder = False, externalObst = True,
                         dis_tol_factor = 10.0):
    # Create KDTree
    tree_obst = KDTree(xObst)
    # Distance from x-array to obstacle
    distances, indices = tree_obst.query(xArray)
    # Give sign to distance if the arrays are in order
    if inOrder:
        # Define polygons
        xCoor = xArray[:, 0]
        yCoor = xArray[:, 1]
        poly = Polygon(zip(xCoor, yCoor))
        xCoor_obst = xObst[:, 0]
        yCoor_obst = xObst[:, 1]
        poly_obst = Polygon(zip(xCoor_obst, yCoor_obst))
        # When the obstacle is outside (x-array nodes must be outside the obstacle)
        if externalObst:
            # Get nodes inside the obstacle
            neg_points = np.array([poly_obst.contains(Point(xi)) for xi in xArray])
            neg_args = np.argwhere(neg_points == True)
        # When obstacle is inside (x-array nodes must be inside the obstacle)
        else:
            # Get nodes outside the obstacle
            neg_points = np.array([not poly_obst.contains(Point(xi)) for xi in xArray])
            neg_args = np.argwhere(neg_points == True)
        if neg_args.size > 0:
            neg_ids = np.hstack(neg_args)
            distances[neg_ids] = -distances[neg_ids]
    contForce = dis_mag*np.exp(-dis_st*(distances - dis_tol))
    # Distance threshold
    less_than_tol_dist_arg = np.argwhere(distances > dis_tol*dis_tol_factor)
    contForce[less_than_tol_dist_arg] = 0.0
    return contForce
# }}}
