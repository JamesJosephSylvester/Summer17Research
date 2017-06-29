def element_setup():
    # Inputs:
    #       p_deg - degree of shape functions
    #       q_deg - degree of quadrature

    # Outputs:
    #       B - mass matrix component
    #       D - stiffness matrix component
    #       q - quadrature points
    #       W - quadrature weights

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

    return B.T, D.T, q, W

def apply_M(u):
    # Inputs:
    #       u - input vector
    #       B - mass matrix component
    #       q - quadrature points
    #       W - quadrature weights
    #       J - Jacobian

    # Outputs:
    #       Mu - Mu = Sum E'B'W|J|BEu

    # Setup
    Mu = np.zeros(pts)

    for i in range(n):
        # Setup
        u_extract = elmt_extract(u, i)

        # Calculate Mu on element
        elmt_result = np.dot(B.T, J * W.dot(B.dot(u_extract)))

        # Insert Result
        Mu += elmt_insert(elmt_result, i)

    # 1D Code
    #for i in range(n):
    #    # Add to Mu
    #    Mu[i * p_deg : (i + 1) * p_deg + 1] += np.dot(B.T, J*W.dot(B.dot(u[i * p_deg : (i + 1) * p_deg + 1])))

    return Mu

def apply_L(u):
    # Inputs:
    #       u - input vector
    #       D - stiffness matrix component
    #       q - quadrature points
    #       W - quadrature weights
    #       J - Jacobian
    #       dX - change of coordinates

    # Outputs:
    #       Lu - Lu = Sum E'D'(dX/dx)'W|J|(dX/dx)DEu

    # Setup
    Lu = np.zeros(pts)

    for i in range(n):
        # Setup
        u_extract = elmt_extract(u, i)

        # Calculate Lu on element
        elmt_result = np.dot(Dx.T, dX*W.dot(J*dX*Dx.dot(u_extract)))
        elmt_result += np.dot(Dy.T, dX*W.dot(J*dX*Dy.dot(u_extract)))

        # Insert Result
        Lu += elmt_insert(elmt_result, i)

    # 1D Code
    #for i in range(n):
    #    # Add to Lu
    #    Lu[i * p_deg : (i + 1) * p_deg + 1] += np.dot(D.T, dX*W.dot(J*dX*D.dot(u[i * p_deg : (i + 1) * p_deg + 1])))

    return Lu

def apply_LHS(u):
    # Apply Operators
    u_return = apply_M(u)

    return u_return

def apply_bc(u):
    # Setup Boundaries
    u_return = u_boundary.copy()

    # Fill Interior
    for i in range(1, pts_x - 1):
        u_return[i * pts_x + 1 : (i + 1) * pts_x - 1] = u[i * pts_x + 1: (i + 1) * pts_x - 1]

    return u_return

def apply_zero_bc(u):
    # Setup Boundaries
    u_return = np.zeros(pts)

    # Fill Interior
    for i in range(1, pts_x - 1):
        u_return[i * pts_x + 1 : (i + 1) * pts_x - 1] = u[i * pts_x + 1 : (i + 1) * pts_x - 1]

    return u_return

def get_bc(u):
    # Setup Boundaries
    u_return = u.copy()

    # Fill Interior
    for i in range(1, pts_x - 1):
        u_return[i * pts_x + 1 : (i + 1) * pts_x - 1] = np.zeros(pts_x - 2)

    return u_return

def elmt_extract(u, elmt):
    # Setup
    col_off = int(np.floor(elmt / nx))
    row_off = int(np.mod(elmt, nx))
    start = col_off * pts_x * p_deg + row_off * p_deg

    # Extract
    element = np.array([u[start + i * pts_x : start + i * pts_x + p_deg + 1] for i in range(p_deg + 1)])

    return element.reshape((p_deg + 1) ** 2)

def elmt_insert(u_e, elmt):
    # Setup
    u_return = np.zeros(pts)

    col_off = int(np.floor(elmt / nx))
    row_off = int(np.mod(elmt, nx))
    start = col_off * pts_x * p_deg + row_off * p_deg

    # Loop through Columns
    for i in range(p_deg + 1):
        u_return[start + i * pts_x : start + i * pts_x + p_deg + 1] = u_e[i * (p_deg + 1) : (i + 1) * (p_deg + 1)]

    return u_return

# Main Code

# Setup
import numpy as np

p_deg = 8                               # degree of shape functions
q_deg = 11                              # degree of quadrature
nx = 4                                  # number of elements in x
n = nx * nx                             # number of elements
pts_x = nx * (p_deg + 1) - (nx - 1)     # number of points in x
pts = pts_x ** 2                        # number of points

I_x = [0, 2 * np.pi]                    # interval
I_y = [0, 2 * np.pi]                    # interval
h = (I_x[1] - I_x[0]) / nx              # element width

u_boundary = np.zeros(pts)              # boundary

def f(x, y):                            # forcing function
    from numpy import tanh, sin, cos
    return sin(2*x) * sin(y)

def f_true(x, y):                       # true function
    return np.sin(2*x) * np.sin(y)

# Setup Element
B, D, q, W = element_setup()
W = np.kron(W, W)
Dx = np.kron(D, B)
Dy = np.kron(B, D)
B = np.kron(B, B)

# Build J, (dX/dx)
# 1D uniform grid
# x = (X + 1) * h / 2 + a
# y = (Y + 1) * h / 2 + a
# dx/dX = h / 2
# dy/dY = h / 2
# |J| = h**2 / 4
dX = 2 / h
J = h ** 2 / 4

# Build Load Vector
Mf = np.zeros(pts)

# Iterate
for i in range(n):
    col_off = int(np.floor(i / nx))
    row_off = int(np.mod(i, nx))

    x = (q + 1) / dX + h * row_off
    y = (q + 1) / dX + h * col_off
    X, Y = np.meshgrid(x, y)
    f_vals = f(X, Y)

    element_Mf = np.dot(B.T, J*W.dot(f_vals.reshape((q_deg ** 2)).T))

    Mf += elmt_insert(element_Mf, i)

# Iterate to solution
# Conjugate gradient
u_old = np.random.rand(pts)
u_old = apply_bc(u_old)

r_temp = Mf - apply_LHS(apply_bc(u_old))
r_old = apply_zero_bc(r_temp) + (get_bc(u_old) - u_boundary)
r_new = r_old.copy()

p_old = r_old.copy()
norm = 1
itr = 1

max_itr = 10*pts

while norm > 10**-16 and itr < max_itr:
    # Calculate new values
    a = np.dot(r_old.T, r_old) / np.dot(p_old.T, apply_LHS(p_old))
    u_new = u_old + a * p_old

    r_temp = Mf - apply_LHS(apply_bc(u_new))
    r_new = apply_zero_bc(r_temp) + (get_bc(u_old) - u_boundary)

    b = np.dot(r_new.T, r_new) / np.dot(r_old.T, r_old)
    p_new = r_new + b * p_old

    # Calculate error
    norm = np.abs(np.max(r_new))

    # Prepare for next iteration
    itr += 1
    u_old = u_new.copy()
    r_old = r_new.copy()
    p_old = p_new.copy()

if itr == max_itr:
    print('max itr reached')

# Plot
import matplotlib.pyplot as plt

x_vals = np.linspace(I_x[0], I_x[1], pts_x)
poly = np.polynomial.legendre.Legendre.basis(p_deg, [-1, 1]).deriv(1)
for i in range(nx):
    x_vals[i * p_deg + 1 : (i + 1) * p_deg] = (poly.roots() + 1) * J + h * i
X, Y = np.meshgrid(x_vals, x_vals)
plt.contourf(X, Y, u_new.reshape(pts_x, pts_x))
plt.show()

# Calculate error
u_true = f_true(X, Y).reshape(pts)

plt.contourf(X, Y, u_true.reshape(pts_x, pts_x))
plt.show()

print('Inf Norm: ' + str(np.max(np.abs(u_true - u_new))))

print(np.max(np.abs(apply_LHS(u_true)) - Mf))

plt.contourf(X, Y, apply_LHS(u_true).reshape(pts_x, pts_x))
plt.show()

plt.contourf(X, Y, Mf.reshape(pts_x, pts_x))
plt.show()
