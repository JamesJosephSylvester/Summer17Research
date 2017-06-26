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
    node = np.linspace(-1, 1, p_deg + 1)
    poly = np.polynomial.legendre.Legendre.basis(p_deg, [-1, 1]).deriv(1)
    node[1 : -1] = poly.roots()

    # B
    B = np.array(
        [np.prod([(q - node[j])/(node[i] - node[j]) for j in range(p_deg + 1) if j != i], 0)
         for i in range(p_deg + 1)])

    # D
    if p_deg == 1:
        D = np.array(
            [np.prod([(q - node[j]) / (node[i] - node[j]) for j in range(p_deg + 1) if j != i], 0)
             *np.sum([1/(q - node[j]) for j in range(p_deg + 1) if j != i], 0)
             for i in range(p_deg + 1)])
        # Smaller expression, but does not allow for odd deg of quadrature
    else:
        D = np.array(
            [np.sum([np.prod([(q - node[j]) / (node[i] - node[j]) for j in range(p_deg + 1) if j != i and j != k], 0)
                      / (node[i] - node[k])
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
        # Add to Mu
        Mu[i * p_deg : (i + 1) * p_deg + 1] += np.dot(B.T, J*W.dot(B.dot(u[i * p_deg : (i + 1) * p_deg + 1])))

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
        # Add to Lu
        Lu[i * p_deg : (i + 1) * p_deg + 1] += np.dot(D.T, dX*W.dot(J*dX*D.dot(u[i * p_deg : (i + 1) * p_deg + 1])))

    return Lu

# Main Code

# Setup
import numpy as np

p_deg = 4                           # degree of shape functions
q_deg = 5                           # degree of quadrature
n = 7                               # number of elements
pts = n * (p_deg + 1) - (n - 1)     # number of points

I = [0, np.pi]                      # interval
h = (I[1] - I[0]) / n               # element width

def f(x):                           # forcing function
    return -np.sin(x)

# Setup Element
B, D, q, W = element_setup()

# Build J, (dX/dx)
# 1D uniform grid
# x = (X + 1) * h / 2 + a
# dx/dX = h / 2
# |J| = h / 2
dX = 2 / h
J = h / 2

# Build Load Vector
Mf = np.zeros(pts)
# Iterate
for i in range(n):
    Mf[i * p_deg : (i + 1) * p_deg + 1] += np.dot(B.T, J*W.dot(f((q + 1) / dX + h * i)))

# Iterate to solution
# Conjugate gradient
u_old = np.array(np.random.rand(pts))
u_old[0] = 0
u_old[-1] = 0
r_old = np.zeros(pts)
r_old[1 : -1] = Mf[1 : -1] - (-apply_L(u_old)[1 : -1])
p_old = r_old.copy()
r_new = r_old.copy()
norm = 1
itr = 1

while norm > 10**-16 and itr < 10*pts:
    # Calculate new values
    a = np.dot(r_old.T, r_old) / np.dot(p_old.T[1 : -1], (-apply_L(p_old)[1 : -1]))
    u_new = u_old + a * p_old
    r_new[1 : -1] = r_old[1 : -1] - a * (-apply_L(p_old)[1 : -1])
    b = np.dot(r_new.T, r_new) / np.dot(r_old.T, r_old)
    p_new = r_new + b * p_old

    # Calculate error
    norm = np.abs(np.max(r_new))

    # Prepate for next iteration
    itr += 1
    u_old = u_new.copy()
    r_old = r_new.copy()
    p_old = p_new.copy()

if itr == 10*pts:
    print('max itr reached')

# Plot
import matplotlib.pyplot as plt

x_vals = np.linspace(I[0], I[1], pts)
poly = np.polynomial.legendre.Legendre.basis(p_deg, [-1, 1]).deriv(1)
for i in range(n):
    x_vals[i * p_deg + 1: (i + 1) * p_deg] = (poly.roots() + 1) * J + h * i
u_true = - f(x_vals)

plt.plot(x_vals, u_true, label = 'True Solution')
plt.plot(x_vals, u_new, label = 'Caluclated Solution')
plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.title('Calculated vs True')
plt.show()

print('Inf Norm: ' + str(np.max(np.abs(u_true - u_new))))