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

    # Boundaries
    # Left
    # Build E
    E = np.zeros((p_deg, pts))
    E[:, :p_deg] = np.eye(p_deg)
    # Add to Mu
    Mu = np.dot(E.T, B[:, 1:].T).dot(W).dot(J).dot(B[:, 1:]).dot(E).dot(u)
    # Right
    # Build E
    E = np.zeros((p_deg, pts))
    E[:, -p_deg:] = np.eye(p_deg)
    # Add to Mu
    Mu += np.dot(E.T, B[:, :-1].T).dot(W).dot(J).dot(B[:, :-1]).dot(E).dot(u)
    # Interior
    for i in range(1, n - 1):
        # Build E
        E = np.zeros((p_deg + 1, pts))

        E[:, i * p_deg - 1 : (i + 1) * p_deg] = np.eye(p_deg + 1)
        # Add to Mu
        Mu += np.dot(E.T, B.T).dot(W).dot(J).dot(B).dot(E).dot(u)

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

    # Boundaries
    # Left
    # Build E
    E = np.zeros((p_deg, pts))
    E[:, :p_deg] = np.eye(p_deg)
    # Add to Lu
    Lu = np.dot(E.T, D[:, 1:].T).dot(dX).dot(W).dot(J).dot(dX).dot(D[:, 1:]).dot(E).dot(u)
    # Right
    # Build E
    E = np.zeros((p_deg, pts))
    E[:, -p_deg:] = np.eye(p_deg)
    # Add to Lu
    Lu += np.dot(E.T, D[:, :-1].T).dot(dX).dot(W).dot(J).dot(dX).dot(D[:, :-1]).dot(E).dot(u)

    # Interior
    for i in range(1, n - 1):
        # Build E
        E = np.zeros((p_deg + 1, pts))
        E[:, i * p_deg - 1 : (i + 1) * p_deg] = np.eye(p_deg + 1)
        # Add to Lu
        Lu += np.dot(E.T, D.T).dot(dX).dot(W).dot(J).dot(dX).dot(D).dot(E).dot(u)

    return Lu

# Main Code

# Setup
import numpy as np

p_deg = 3                           # degree of shape functions
q_deg = 5                           # degree of quadrature
n = 3                               # number of elements
pts = n * (p_deg + 1) - (n - 1) - 2 # number of points

I = [0, np.pi]                  # interval
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

# Boundaries
# Left
E = np.zeros((p_deg, pts))
E[:, :p_deg] = np.eye(p_deg)
# Add to Mf
Mf = np.dot(E.T, B[:, 1:].T).dot(W).dot(J).dot(np.diag(f((q + 1) / dX))).dot(B[:, 1:]).dot(E).dot(np.ones(pts))
# Right
# Build E
E = np.zeros((p_deg, pts))
E[:, -p_deg:] = np.eye(p_deg)
# Add to Mf
Mf += np.dot(E.T, B[:, :-1].T).dot(W).dot(J).dot(np.diag(f((q + 1) / dX + h * (n - 1)))).dot(B[:, :-1]).dot(E).dot(np.ones(pts))
print(f((q + 1) / dX))
# Interior
for i in range(1, n - 1):
    # Build E
    E = np.zeros((p_deg + 1, pts))
    E[:, i * p_deg - 1: (i + 1) * p_deg] = np.eye(p_deg + 1)
    Mf += np.dot(E.T, B.T).dot(W).dot(J).dot(np.diag(f((q + 1) / dX + h * i))).dot(B).dot(E).dot(np.ones(pts))

# Iterate to solution
# Conjugate gradient
u_old = np.array(np.random.rand(pts))
r_old = Mf - (-apply_L(u_old))
p_old = r_old.copy()
norm = 1
itr = 1

while norm > 10**-16 and itr < 10*pts:
    # Calculate new values
    a = np.dot(r_old.T, r_old) / np.dot(p_old.T, (-apply_L(p_old)))
    u_new = u_old + a * p_old
    r_new = r_old - a * (-apply_L(p_old))
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
x_vals = np.linspace(I[0] + h / p_deg, I[1] - h / p_deg, pts)
u_true = - f(x_vals)
plt.plot(x_vals, u_true)
plt.plot(x_vals, u_new)
plt.show()
