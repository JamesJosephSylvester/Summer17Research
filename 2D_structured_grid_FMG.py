# 1D element setup
def element_setup(p_deg, q_deg):
    # Inputs:
    #       p_deg - degree of shape functions
    #       q_deg - degree of quadrature

    # Outputs:
    #       B - mass matrix component
    #       D - stiffness matrix component
    #       q - quadrature points
    #       W - quadrature weights
    #       nodes - basis nodes

    # Quadrature Points and Weights
    q, w = np.polynomial.legendre.leggauss(q_deg)
    W = np.diag(w)

    # B, D matrices
    nodes = np.linspace(-1, 1, p_deg + 1)
    poly = np.polynomial.legendre.Legendre.basis(p_deg, [-1, 1]).deriv(1)
    nodes[1 : -1] = poly.roots()

    # B
    B = np.array(
        [np.prod([(q - nodes[j])/(nodes[i] - nodes[j]) for j in range(p_deg + 1) if j != i], 0)
         for i in range(p_deg + 1)])

    # D
    if p_deg == 1:
        D = np.array(
            [np.prod([(q - nodes[j]) / (nodes[i] - nodes[j]) for j in range(p_deg + 1) if j != i], 0)
             *np.sum([1/(q - nodes[j]) for j in range(p_deg + 1) if j != i], 0)
             for i in range(p_deg + 1)])
        # Smaller expression, but does not allow for odd deg of quadrature
    else:
        D = np.array(
            [np.sum([np.prod([(q - nodes[j]) / (nodes[i] - nodes[j]) for j in range(p_deg + 1) if j != i and j != k], 0)
                      / (nodes[i] - nodes[k])
             for k in range(p_deg + 1) if k != i], 0)
             for i in range(p_deg + 1)])

    return B.T, D.T, q, W, nodes

# Setup computation
def fn_forcing(x, y):                   # forcing function
    return np.sin(2 * x) * np.sin(y) + x + y #- 5 * sin(2 * x) * sin(y)

def fn_true(x, y):                      # true function
    return np.sin(2 * x) * np.sin(y) + x + y

def f_setup(op_mats, op_par, grid):
    # Inputs
    #       op_mat - Operator matrices
    #       op_par - Operator parameters
    #       grid - grid information

    # Outputs
    #       f_vec - load vector

    # Setup
    f_vec = np.zeros(grid['pts'])

    # Iterate
    for i in range(grid['n']):
        row = int(np.floor(i / grid['n_x']))
        col = int(np.mod(i, grid['n_x']))

        x = (op_mats['q'] + 1) / op_par['dX'] + col / op_par['dX'] * 2
        y = (op_mats['q'] + 1) / op_par['dX'] + row / op_par['dX'] * 2
        X, Y = np.meshgrid(x, y)
        f_vals = fn_forcing(X, Y).reshape((grid['q_deg'] ** 2))

        element_f_vec = np.dot(op_mats['B'].T, op_par['J'] * op_mats['W'].dot(f_vals.T))

        f_vec += elmt_insert(element_f_vec, i, grid)

    return f_vec

def bc_setup(op_par, grid):
    # Inputs
    #       op_pars - operator parameters
    #       grid - grid information

    # Outputs
    #       u_boundary - boundary values

    # Setup
    x = np.linspace(I_x[0], I_x[1], grid['pts_x'])
    poly = np.polynomial.legendre.Legendre.basis(grid['p_deg'], [-1, 1]).deriv(1)

    # Loop
    for i in range(grid['n_x']):
        x[i * grid['p_deg'] + 1: (i + 1) * grid['p_deg']] = (poly.roots() + 1) / op_par['dX'] + i / op_par['dX'] * 2

    X, Y = np.meshgrid(x, x)

    u_boundary = (X + Y).reshape(grid['pts'])
    u_boundary = get_bc(u_boundary, grid)

    return u_boundary


# Apply operators
def apply_M(u, op_mats, op_par, grid):
    # Inputs:
    #       u - input vector
    #       op - operator components
    #       grid - grid details

    # Outputs:
    #       Mu - Mu = Sum E'B'W|J|BEu

    # Setup
    Mu = np.zeros(grid['pts'])

    for i in range(grid['n']):
        # Setup
        u_extract = elmt_extract(u, i, grid)

        # Calculate Mu on element
        elmt_result = np.dot(op_mats['B'].T, op_par['J'] * op_mats['W'].dot(op_mats['B'].dot(u_extract)))

        # Insert Result
        Mu += elmt_insert(elmt_result, i, grid)

    # 1D Code
    #for i in range(n):
    #    # Add to Mu
    #    Mu[i * p_deg : (i + 1) * p_deg + 1] += np.dot(B.T, J*W.dot(B.dot(u[i * p_deg : (i + 1) * p_deg + 1])))

    return Mu

def apply_L(u, op_mats, op_par, grid):
    # Inputs:
    #       u - input vector
    #       op - operator components
    #       grid - grid details

    # Outputs:
    #       Lu - Lu = Sum E'D'(dX/dx)'W|J|(dX/dx)DEu

    # Setup
    Lu = np.zeros(grid['pts'])

    for i in range(grid['n']):
        # Setup
        u_extract = elmt_extract(u, i, grid)

        # Calculate Lu on element
        elmt_result = np.dot(op_mats['Dx'].T, op_par['dX'] * op_mats['W'].dot(op_par['J'] * op_par['dX'] * op_mats['Dx'].dot(u_extract)))
        elmt_result += np.dot(op_mats['Dy'].T, op_par['dX'] * op_mats['W'].dot(op_par['J'] * op_par['dX'] * op_mats['Dy'].dot(u_extract)))

        # Insert Result
        Lu += elmt_insert(elmt_result, i, grid)

    # 1D Code
    #for i in range(n):
    #    # Add to Lu
    #    Lu[i * p_deg : (i + 1) * p_deg + 1] += np.dot(D.T, dX*W.dot(J*dX*D.dot(u[i * p_deg : (i + 1) * p_deg + 1])))

    return Lu

def apply_op(u, op_mats, op_par, grid):
    # Inputs:
    #       u - input vector
    #       op - operator components
    #       grid - grid details

    # Outputs:
    #       u_return - action of LHS on u

    # Apply Operators
    u_return = apply_M(u, op_mats, op_par, grid)

    return u_return


# Boundary conditions
def apply_bc(u, grid, bc):
    # Inputs:
    #       u - input vector

    # Outputs:
    #       u_return - u with b/c applied

    # Setup Boundaries
    u_return = bc.copy()

    # Fill Interior
    for i in range(1, grid['pts_x'] - 1):
        u_return[i * grid['pts_x'] + 1 : (i + 1) * grid['pts_x'] - 1] = u[i * grid['pts_x'] + 1: (i + 1) * grid['pts_x'] - 1]

    return u_return

def apply_zero_bc(u, grid):
    # Inputs:
    #       u - input vector
    #       grid - grid details

    # Outputs:
    #       u_return - u with zero b/c applied

    # Setup Boundaries
    u_return = np.zeros(grid['pts'])

    # Fill Interior
    for i in range(1, grid['pts_x'] - 1):
        u_return[i * grid['pts_x'] + 1 : (i + 1) * grid['pts_x'] - 1] = u[i * grid['pts_x'] + 1 : (i + 1) * grid['pts_x'] - 1]

    return u_return

def get_bc(u, grid):
    # Inputs:
    #       u - input vector
    #       grid - grid details

    # Outputs:
    #       u_return - boundary values of u

    # Setup Boundaries
    u_return = u.copy()

    # Fill Interior
    for i in range(1, grid['pts_x'] - 1):
        u_return[i * grid['pts_x'] + 1 : (i + 1) * grid['pts_x'] - 1] = np.zeros(grid['pts_x'] - 2)

    return u_return


# Element handling
def elmt_extract(u, elmt, grid):
    # Inputs:
    #       u - input vector
    #       elmt - number of element to extract
    #       grid - grid details

    # Outputs:
    #       element - element extracted

    # Setup
    row_off = int(np.floor(elmt / grid['n_x'])) * grid['p_deg']
    col_off = int(np.mod(elmt, grid['n_x'])) * grid['p_deg']
    start = row_off * grid['pts_x'] + col_off

    # Extract
    element = np.array([u[start + i * grid['pts_x'] : start + i * grid['pts_x'] + (grid['p_deg'] + 1)] for i in range(grid['p_deg'] + 1)])

    return element.reshape((grid['p_deg'] + 1) ** 2)

def elmt_insert(u_e, elmt, grid):
    # Inputs:
    #       u_e - input vector
    #       elmt - number of element to insert
    #       grid - grid details

    # Outputs:
    #       u_return - element inserted

    # Setup
    u_return = np.zeros(grid['pts'])
    row_off = int(np.floor(elmt / grid['n_x'])) * grid['p_deg']
    col_off = int(np.mod(elmt, grid['n_x'])) * grid['p_deg']
    start = row_off * grid['pts_x'] + col_off

    # Loop through Columns
    for i in range(grid['p_deg'] + 1):
        u_return[start + i * grid['pts_x'] : start + i * grid['pts_x'] + (grid['p_deg'] + 1)] = u_e[i * (grid['p_deg'] + 1) : (i + 1) * (grid['p_deg'] + 1)]

    return u_return


# CG solve
def cg_solve(u_0, f_vec, max_itr, tol, op_mats, op_par, grid, bc):
    # Inputs:
    #       u_0 - initial guess
    #       f_vec - load vector
    #       max_itr - maximum number of iterations
    #       tol - error tolerance
    #       op - operator components
    #       grid - grid details

    # Outputs:
    #       u - solution of LHS(u) = f_vec

    # Setup
    u = u_0
    u = apply_bc(u, grid, bc)

    r_temp = f_vec - apply_op(apply_bc(u, grid, bc), op_mats, op_par, grid)
    r_old = apply_zero_bc(r_temp, grid) + (get_bc(u, grid) - bc)
    r_new = r_old.copy()

    p = r_old.copy()
    norm = 1
    itr = 1

    while norm > tol and itr < max_itr:
        # Calculate new values
        a = np.dot(r_old.T, r_old) / np.dot(p.T, apply_op(p, op_mats, op_par, grid))
        u = u + a * p

        r_temp = f_vec - apply_op(apply_bc(u, grid, bc), op_mats, op_par, grid)
        r_new = apply_zero_bc(r_temp, grid) + (get_bc(u, grid) - bc)

        b = np.dot(r_new.T, r_new) / np.dot(r_old.T, r_old)
        p = r_new + b * p

        # Calculate error
        norm = np.abs(np.max(r_new))

        # Early exit conditions
        if norm > 500:
            itr = max_itr + 1
        if np.max(np.abs(r_old - r_new)) < tol / 10:
            itr = max_itr

        # Prepare for next iteration
        itr += 1
        r_old = r_new.copy()

    # Check exit conditions
    if itr == max_itr:
        print('CG max itr reached')
    elif itr == max_itr + 1:
        print('CG iterates not changing')
    elif itr == max_itr + 2:
        print('CG solution diverged')
    else:
        print('CG residual within tolerance')
        print(str(itr) + ' iterations')

    return u


# FMG solve
def fmg_solve(f_vec, levels, tol, op_mats, op_par, grid, bc):
    # Inputs:
    #       f_vec - load vector
    #       levels - number of levels
    #       tol - error tolerance
    #       op - operator components
    #       grid - grid details

    # Outputs:
    #       u - solution of LHS(u) = f_vec

    # Setup
    smooths = 3
    omega = 2 / 3
    I = np.kron(np.array([[1, 0.5, 0], [0, 0.5, 1]]), np.array([[1, 0.5, 0], [0, 0.5, 1]])).T

    # Setup grid arrays
    f_vecs = [[] for i in range(levels)]
    bc_vecs = [[] for i in range(levels)]
    op_pars = [[] for i in range(levels)]
    grids = [[] for i in range(levels)]

    # Linearize
    f_vecs[levels - 1], op_mats_lin, op_pars[levels - 1], grids[levels - 1] = p_to_lin(grid)
    bc_vecs[levels - 1] = bc_setup(op_pars[levels - 1], grids[levels - 1])
    grids[levels - 1]['I'] = I


    # Set up f_vecs, ops, grids
    for i in range(1, levels):
        op_pars[levels - i - 1], grids[levels - i - 1] = restrict_op_grid(op_mats_lin, op_pars[levels - i], grids[levels - i])
        f_vecs[levels - i - 1] = f_setup(op_mats_lin, op_pars[levels - i - 1], grids[levels - i - 1])
        bc_vecs[levels - i - 1] = bc_setup(op_pars[levels - i - 1], grids[levels - i - 1])

    # Initial guess
    u = np.random.rand(grids[0]['pts'])

    # Loop through V-cycles at levels
    for i in range(levels):
        print('Level ' + str(i))
        # Prolongate
        if i:
            u = prolongate_u(u, grids[i], grids[i - 1])
            u = apply_bc(u, grids[i], bc_vecs[i])

        # Presmooth
        for j in range(smooths):
            u = u - apply_zero_bc(omega * (apply_op(u, op_mats_lin, op_pars[i], grids[i]) - f_vecs[i]), grids[i]) * op_pars[i]['diag_lumped_inv']

        # Solve if coarsest
        if not i:
            # Level 0 solve
            u = cg_solve(u, f_vecs[0], 10 * grids[0]['pts'], tol * 10 ** 3, op_mats_lin, op_pars[0], grids[0], bc_vecs[0])
        # Otherwise restrict, V Cycle, and correct
        else:
            # Restrict
            u_restrict = restrict_u(u, grids[i], grids[i - 1])
            u_restrict = apply_bc(u_restrict, grids[i - 1], bc_vecs[i - 1])
            # V cycle
            u_new = v_cycle(u_restrict, f_vecs, i, op_mats_lin, op_pars, grids, bc_vecs)
            # Correct
            u = u + prolongate_u(apply_zero_bc(u_new - u_restrict, grids[i - 1]), grids[i], grids[i - 1])

        # Postsmooth
        for j in range(smooths):
            u = u - apply_zero_bc(omega * (apply_op(u, op_mats_lin, op_pars[i], grids[i]) - f_vecs[i]), grids[i]) * op_pars[i]['diag_lumped_inv']

    print('Level ' + str(levels))
    # Interpolate to p
    u = u_to_p(u, op_mats, grid, grids[levels - 1])
    u = apply_bc(u, grid, bc)

    # Presmooth
    for smooths in range(3 * smooths):
        u = u - apply_zero_bc(omega * (apply_op(u, op_mats, op_par, grid) - f_vec), grid) * op_par['diag_lumped_inv']

    # Restrict
    u_restrict = u_to_lin(u, op_mats, grid, grids[levels - 1])
    # V cycle
    u_new = v_cycle(u_restrict, f_vecs, levels, op_mats_lin, op_pars, grids, bc_vecs)
    # Correct
    u = u + u_to_p(apply_zero_bc(u_new - u_restrict, grids[levels - 1]), op_mats, grid, grids[levels - 1])

    # Postmooth
    for smooths in range(3 * smooths):
        u = u - apply_zero_bc(omega * (apply_op(u, op_mats, op_par, grid) - f_vec), grid) * op_par['diag_lumped_inv']

    return u

def v_cycle(u, f_vecs, levels, op_mats, op_pars, grids, bc_vecs):
    smooths = 1
    omega = 2 / 3
    i = levels - 1
    # Presmooth
    for j in range(smooths):
        u = u - apply_zero_bc(omega * (apply_op(u, op_mats, op_pars[i], grids[i]) - f_vecs[i]), grids[i]) * op_pars[i]['diag_lumped_inv']

    # Solve if coarsest
    if not i:
        u = cg_solve(u, f_vecs[0], 10 * grids[0]['pts'], tol, op_mats, op_pars[0], grids[0], bc_vecs[0])
    # Otherwise restrict, V Cycle, and correct
    else:
        # Restrict
        u_restrict = restrict_u(u, grids[i], grids[i - 1])
        u_restrict = apply_bc(u_restrict, grids[i - 1], bc_vecs[i - 1])
        # V cycle
        u_new = v_cycle(u_restrict, f_vecs, i, op_mats, op_pars, grids, bc_vecs)
        # Correct
        u = u + prolongate_u(apply_zero_bc(u_new - u_restrict, grids[i - 1]), grids[i], grids[i - 1])

    # Postsmooth
    for j in range(smooths):
        u = u - apply_zero_bc(omega * (apply_op(u, op_mats, op_pars[i], grids[i]) - f_vecs[i]), grids[i]) * op_pars[i]['diag_lumped_inv']

    return u

def p_to_lin(grid):
    # Inputs
    #   grid - grid with p = p

    # Outputs
    #   f_vec - load vector at linear resolution
    #   op - operator with p = 1
    #   grid - grid with p = 1

    # Linear setup
    B_lin, D_lin, q_lin, W_lin, nodes = element_setup(1, grid['q_deg'])
    Dy_lin = np.kron(D_lin, B_lin)
    Dx_lin = np.kron(B_lin, D_lin)
    W_lin = np.kron(W_lin, W_lin)
    B_lin = np.kron(B_lin, B_lin)
    pts_x_lin = grid['n_x'] + 1
    pts_lin = pts_x_lin ** 2

    grid_lin = {'p_deg': 1, 'q_deg': grid['q_deg'], 'n': grid['n'], 'n_x': grid['n_x'], 'pts': pts_lin, 'pts_x': pts_x_lin}
    op_mats_lin = {'q': q, 'B': B_lin, 'Dx': Dx_lin, 'Dy': Dy_lin, 'W': W_lin}
    op_par_lin = {'J': op_par['J'], 'dX': op_par['dX']}
    ones_vec = np.ones(grid_lin['pts'])
    op_par_lin['diag_lumped_inv'] = ones_vec / apply_op(ones_vec, op_mats_lin, op_par_lin, grid_lin)

    f_vec_lin = f_setup(op_mats_lin, op_par_lin, grid_lin)

    return f_vec_lin, op_mats_lin, op_par_lin, grid_lin

def u_to_p(u, op_mats, grid, grid_lin):
    # Inputs:
    #       u - input vector
    #       op_mats - op details
    #       grid - grid details
    #       grid_lin - linear grid details

    # Outputs:
    #       u_return - interpolation of element with p basis

    # Setup
    u_return = np.zeros(grid['pts'])
    pts_side = grid['p_deg'] + 1
    I = np.array([1 - (op_mats['nodes'] + 1)/ 2, (op_mats['nodes'] + 1) / 2]).T
    I = np.kron(I, I)

    # Loop
    for i in range(grid['n']):
        extract = elmt_extract(u, i, grid_lin)
        insert = np.matmul(I, extract)
        insert = elmt_insert(insert, i, grid)
        u_return = np.array([u_return[j] if u_return[j] else insert[j] for j in range(grid['pts'])])

    return u_return

def u_to_lin(u, op_mats, grid, grid_lin):
    # Inputs:
    #       u - input vector
    #       op_mats - operator details
    #       grid - grid details
    #       grid_lin - linear grid details

    # Outputs:
    #       u_return - sampling of element with linear basis

    # Setup
    u_return = np.zeros(grid_lin['pts'])
    pts_side = grid['p_deg'] + 1
    I = np.array([1 - (op_mats['nodes'] + 1)/ 2, (op_mats['nodes'] + 1) / 2]).T
    I = np.kron(I, I)

    # Loop
    for i in range(grid['n']):
        extract = elmt_extract(u, i, grid)
        insert = np.matmul(I.T, extract)
        insert = elmt_insert(insert, i, grid_lin)
        u_return = np.array([u_return[j] if u_return[j] else insert[j] for j in range(grid_lin['pts'])])

    return u_return

def restrict_op_grid(op_mats, op_fine, grid_fine):
    # Input
    #   op_fine - fine operator
    #   grid_fine - fine grid

    # Outputs
    #   op_coarse - coarse operator
    #   grid_coarse - coarse grid

    # Grid
    grid_coarse = grid_fine.copy()
    grid_coarse['n_x'] = int(grid_coarse['n_x'] / 2)
    grid_coarse['n'] = int(grid_coarse['n_x'] ** 2)
    grid_coarse['pts_x'] = int(grid_coarse['n_x'] + 1)
    grid_coarse['pts'] = int(grid_coarse['pts_x'] ** 2)

    # Operator
    op_coarse = op_fine.copy()
    op_coarse['dX'] *= 2
    op_coarse['J'] /= 2 ** 2
    ones_vec = np.ones(grid_coarse['pts'])
    op_coarse['diag_lumped_inv'] = ones_vec / apply_op(ones_vec, op_mats, op_coarse, grid_coarse)

    return op_coarse, grid_coarse

def restrict_u(u, grid_fine, grid_coarse):
    # Inputs
    #   u - fine resolution
    #   grid_fine - fine grid
    #   grid_coarse - coarse grid

    # Outputs
    #   u_return - coarse resolution

    # Setup
    u_return = np.zeros(grid_coarse['pts'])

    # Loop through coarse elements
    for i in range(grid_coarse['n']):
        row_off = int(np.floor(i / grid_coarse['n_x']))
        col_off = int(np.mod(i, grid_coarse['n_x']))
        start = 2 * row_off * grid_fine['n_x'] + 2 * col_off

        u_e0 = elmt_extract(u, start, grid_fine)
        u_e1 = elmt_extract(u, start + 1, grid_fine)
        u_e2 = elmt_extract(u, start + grid_fine['n_x'], grid_fine)
        u_e3 = elmt_extract(u, start + grid_fine['n_x'] + 1, grid_fine)
        insert = np.array([u_e0[0], u_e0[1], u_e1[1], u_e0[2], u_e0[3], u_e1[3], u_e2[2], u_e2[3], u_e3[3]])
        insert = np.matmul(grid_coarse['I'].T, insert)
        insert = elmt_insert(insert, i, grid_coarse)

        u_return += insert

    return u_return

def prolongate_u(u, grid_fine, grid_coarse):
    # Inputs
    #   u - coarse resolution
    #   grid_fine - fine grid
    #   grid_coarse - coarse grid

    # Outputs
    #   u_return - fine resolution

    # Setup
    u_return = np.zeros(grid_fine['pts'])
    grid_false = grid_coarse.copy()
    grid_false['p_deg'] = 2
    grid_false['pts_x'] = grid_fine['pts_x']
    grid_false['pts'] = grid_fine['pts']

    # Loop through coarse elements
    for i in range(grid_coarse['n']):
        extract = elmt_extract(u, i, grid_coarse)
        insert = np.matmul(grid_coarse['I'], extract)
        insert = elmt_insert(insert, i, grid_false)

        u_return = np.array([u_return[j] if u_return[j] else insert[j] for j in range(grid_fine['pts'])])

    return u_return


# Main Code

# Setup
import numpy as np
import time
import matplotlib.pyplot as plt

p_deg = 3                               # degree of shape functions
q_deg = 4                               # degree of quadrature
levels = 3                              # number of muligrid levels
n_x = 9 * 2 ** (levels - 1)             # number of elements in x
n = n_x * n_x                           # number of elements
pts_x = n_x * (p_deg + 1) - (n_x - 1)   # number of points in x
pts = pts_x ** 2                        # number of points

I_x = [0, 2 * np.pi]                    # interval
I_y = [0, 2 * np.pi]                    # interval
h = (I_x[1] - I_x[0]) / n_x             # element width

# Setup Element
B, D, q, W, nodes = element_setup(p_deg, q_deg)
Dy = np.kron(D, B)
Dx = np.kron(B, D)
W = np.kron(W, W)
B = np.kron(B, B)

# Build J, (dX/dx)
J = h ** 2 / 4
dX = 2 / h

# Dictionaries
grid = {'p_deg': p_deg, 'q_deg': q_deg, 'n': n, 'n_x': n_x, 'pts': pts, 'pts_x': pts_x}
op_mats = {'q': q, 'B': B, 'Dx': Dx, 'Dy': Dy, 'W': W, 'nodes': nodes}
op_par = {'J': J, 'dX': dX}
ones_vec = np.ones(grid['pts'])
op_par['diag_lumped_inv'] = ones_vec / apply_op(ones_vec, op_mats, op_par, grid)


# Boundary
u_boundary = bc_setup(op_par, grid)

# Build Load Vector
f_vec = f_setup(op_mats, op_par, grid)


# Calculate true solution
x = np.linspace(I_x[0], I_x[1], pts_x)
poly = np.polynomial.legendre.Legendre.basis(p_deg, [-1, 1]).deriv(1)
for i in range(n_x):
    x[i * grid['p_deg'] + 1 : (i + 1) * grid['p_deg']] = (poly.roots() + 1) / op_par['dX'] + h * i
X, Y = np.meshgrid(x, x)
u_true = fn_true(X, Y).reshape(pts)


# Solvers
max_itr = 10 * grid['pts']
tol = 10 ** -12

# Conjugate Gradient solve
print('Conjugate Gradient')
u_0 = np.random.rand(grid['pts'])
tic = time.time()
u_cg = cg_solve(u_0, f_vec, max_itr, tol, op_mats, op_par, grid, u_boundary)
toc = time.time()
print('Inf Norm: ' + str(np.max(np.abs(u_true - u_cg))))
print('Time: ' + str(toc - tic))

# FMG solve
print('\n')
print('Full Multigrid')
tic = time.time()
u_fmg = fmg_solve(f_vec, levels, tol * 10 ** 3, op_mats, op_par, grid, u_boundary)
toc = time.time()
print('Inf Norm: ' + str(np.max(np.abs(u_true - u_fmg))))
print('Time: ' + str(toc - tic))


# Plot
fig, axarr = plt.subplots(2, 2)
axarr[0, 0].contourf(X, Y, u_cg.reshape(grid['pts_x'], grid['pts_x']))
axarr[0, 0].title.set_text('CG Result')
axarr[0, 1].contourf(X, Y, u_fmg.reshape(grid['pts_x'], grid['pts_x']))
axarr[0, 1].title.set_text('FMG Result')
axarr[1, 0].contourf(X, Y, (u_true - u_cg).reshape(grid['pts_x'], grid['pts_x']))
axarr[1, 0].title.set_text('CG Error')
axarr[1, 1].contourf(X, Y, (u_true - u_fmg).reshape(grid['pts_x'], grid['pts_x']))
axarr[1, 1].title.set_text('FMG Error')
plt.show()
