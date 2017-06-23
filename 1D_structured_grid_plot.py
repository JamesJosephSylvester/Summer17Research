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
        Lu += np.dot(E.T, D.T.dot(dX*W.dot(J*dX*D.dot(E.dot(u)))))

    return Lu

# Main Code
# Setup
import numpy as np
import matplotlib.pyplot as plt
p_degs = range(2, 7)
run_errs = np.zeros((max(p_degs)+1, 7))
run_delx = run_errs.copy()

q_deg = 8       # degree of quadrature
def f(x):       # forcing function
    #return (18 * np.pi**2 - 36 * np.pi * x + 12 * x**2) / 500
    from numpy import tanh, sin, cos
    return (2*(tanh(x)**2 - 1)*sin(x)*tanh(x) - 2*(tanh(x)**2 - 1)*cos(x) - sin(x)*tanh(x))

def f_true(x):  # true function
    #return (x**2 * (x - 3 * np.pi)**2) / 500
    return np.tanh(x)*np.sin(x)

# Run FE with different p_deg, n
for p_deg in p_degs:
    for n in range(10, 70, 10):
        # Setup
        pts = n * (p_deg + 1) - (n - 1) - 2 # number of points

        I = [0, 3 * np.pi]                  # interval
        h = (I[1] - I[0]) / n               # element width

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
        Mf = np.dot(E.T, B[:, 1:].T).dot(W).dot(J).dot(f((q + 1) / dX))
        # Right
        # Build E
        E = np.zeros((p_deg, pts))
        E[:, -p_deg:] = np.eye(p_deg)
        # Add to Mf
        Mf += np.dot(E.T, B[:, :-1].T).dot(W).dot(J).dot(f((q + 1) / dX + h * (n - 1)))

        # Interior
        for i in range(1, n - 1):
            # Build E
            E = np.zeros((p_deg + 1, pts))
            E[:, i * p_deg - 1: (i + 1) * p_deg] = np.eye(p_deg + 1)
            Mf += np.dot(E.T, B.T.dot(W.dot(J*f((q + 1) / dX + h * i))))

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

        # Calculate error
        x_vals = np.linspace(I[0] + h / p_deg, I[1] - h / p_deg, pts)
        u_true = f_true(x_vals)
        norm = np.max(np.abs(u_true - u_new))

        run_errs[p_deg, int(n / 10)] = norm
        run_delx[p_deg, int(n / 10)] = h / (p_deg + 1)

# Plot
symbols = iter(['o', '^', 's', '*', '>', '+'])
for p_deg in p_degs:
    plt.loglog(run_delx[p_deg, :], run_errs[p_deg, :], next(symbols), label = 'p = '+str(p_deg))
    plt.loglog(run_delx[p_deg, :], run_delx[p_deg, :]**(p_deg+1), label='$h^{%d}$'%(p_deg+1))
plt.legend()
plt.xlabel('dx')
plt.ylabel('||error||')
plt.title('Error vs dx')
plt.show()

# Plot last result
plt.plot(x_vals, u_true, label = 'True Solution')
plt.plot(x_vals, u_new, label = 'Calculated Solution')
plt.legend()
plt.xlabel('x')
plt.ylabel('u')
plt.title('Calculated vs True')
plt.show()
