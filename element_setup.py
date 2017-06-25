def element_setup(p_deg, q_deg):
    # Inputs:
    #       p_deg - degree of shape functions
    #       q_deg - degree of quadrature

    # Outputs:
    #       B - mass matrix component
    #       D - stiffness matrix component
    #       q - quadrature points
    #       W - quadrature weights

    # Setup
    import numpy as np

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


# Check Output
B, D, q, W = element_setup(2, 100)

print("q")
print(q)

print("W")
print(W)

print("B")
print(B)

print("D")
print(D)

import numpy as np
print("B' W B")
print(np.matmul(B.T, np.matmul(W, B)))

print("D' W D")
print(np.matmul(D.T, np.matmul(W, D)))

# Plots
import matplotlib.pyplot as plt
for i in range(3):
    plt.plot(q, B[:, i])
plt.show()

for i in range(3):
    plt.plot(q, D[:, i])
plt.show()
