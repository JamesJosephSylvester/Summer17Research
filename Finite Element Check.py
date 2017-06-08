# Fixed Point Iteration
def fixed_pt(A, b, u_0):
    # Setup
    length = len(u_0)
    u_old = numpy.copy(u_0)
    u_new = numpy.copy(u_0)
    for i in range(1, length):
        u_new[i] = (b[i] - numpy.dot(A[i, :], u_new.T) + A[i, i] * u_new[i]) / A[i, i]

    # Iterate
    while max(abs(u_old - u_new)) > 10 ** (-15):
        u_old = numpy.copy(u_new)
        for i in range(1, length):
            u_new[i] = (b[i] - numpy.dot(A[i, :], u_new.T) + A[i, i] * u_new[i]) / A[i, i]

    # Return
    return u_new


# U function
def u(x, y):
    return numpy.sin(x * numpy.pi) * numpy.sin(y * numpy.pi)


# U True
def u_true(mx):
    # Setup
    u_true_vals = numpy.zeros(mx * mx)
    # Loop x
    for x in range(1, mx + 1):
        # Loop y
        for y in range(1, mx + 1):
            u_true_vals[(y - 1) + (x - 1) * mx] = u(x / (mx + 1), y / (mx + 1))

    return u_true_vals


# Phi Functions
# Phi_1
def phi_1(x, y, x_0, y_0):
    return (1 - (mx + 1) ** 2 * (x - x_0) ** 2) * (1 - (mx + 1) ** 2 * (y - y_0) ** 2)


def phi_2_y(x, y, x_0, y_0):
    if y >= y_0:
        return (1 - (mx + 1) ** 2 * (x - x_0) ** 2) * (
        1 - 3 / 2 * (mx + 1) * (y - y_0) + 1 / 2 * (mx + 1) ** 2 * (y - y_0) ** 2)
    else:
        return (1 - (mx + 1) ** 2 * (x - x_0) ** 2) * (
        1 + 3 / 2 * (mx + 1) * (y - y_0) + 1 / 2 * (mx + 1) ** 2 * (y - y_0) ** 2)


def phi_2_x(x, y, x_0, y_0):
    if x >= x_0:
        return (1 - (mx + 1) ** 2 * (y - y_0) ** 2) * (
        1 - 3 / 2 * (mx + 1) * (x - x_0) + 1 / 2 * (mx + 1) ** 2 * (x - x_0) ** 2)
    else:
        return (1 - (mx + 1) ** 2 * (y - y_0) ** 2) * (
        1 + 3 / 2 * (mx + 1) * (x - x_0) + 1 / 2 * (mx + 1) ** 2 * (x - x_0) ** 2)


def phi_4(x, y, x_0, y_0):
    if x >= x_0 and y >= y_0:
        return (1 - 3 / 2 * (mx + 1) * (x - x_0) + 1 / 2 * (mx + 1) ** 2 * (x - x_0) ** 2) * (
        1 - 3 / 2 * (mx + 1) * (y - y_0) + 1 / 2 * (mx + 1) ** 2 * (y - y_0) ** 2)
    elif x >= x_0:
        return (1 - 3 / 2 * (mx + 1) * (x - x_0) + 1 / 2 * (mx + 1) ** 2 * (x - x_0) ** 2) * (
        1 + 3 / 2 * (mx + 1) * (y - y_0) + 1 / 2 * (mx + 1) ** 2 * (y - y_0) ** 2)
    elif y >= y_0:
        return (1 + 3 / 2 * (mx + 1) * (x - x_0) + 1 / 2 * (mx + 1) ** 2 * (x - x_0) ** 2) * (
        1 - 3 / 2 * (mx + 1) * (y - y_0) + 1 / 2 * (mx + 1) ** 2 * (y - y_0) ** 2)
    else:
        return (1 + 3 / 2 * (mx + 1) * (x - x_0) + 1 / 2 * (mx + 1) ** 2 * (x - x_0) ** 2) * (
        1 + 3 / 2 * (mx + 1) * (y - y_0) + 1 / 2 * (mx + 1) ** 2 * (y - y_0) ** 2)


# Forcing Function
def force(x, y):
    return -2 * (numpy.pi ** 2) * numpy.sin(x * numpy.pi) * numpy.sin(y * numpy.pi)


# Integrate
def integrate(element, x_0, y_0, hx, hy):
    deg = 128
    vals, wgts = numpy.polynomial.legendre.leggauss(deg)
    shift_vals = (vals + 1)*hy + (y_0 - hy)
    integral = 0
    if element == "phi_1":
        for k in range(deg):
            integral += wgts[k] * sum(
                wgts * [phi_1(x_0 - hx + (vals[k] + 1)*hx, y_val, x_0, y_0) for y_val in shift_vals] * force(
                    x_0 - hx + (vals[k] + 1)*hx, shift_vals))
        return integral * hx * hy
    elif element == "phi_2_x":
        for k in range(deg):
            integral += wgts[k] * sum(
                wgts * [phi_2_x(x_0 - hx + (vals[k] + 1)*hx, y_val, x_0, y_0) for y_val in shift_vals] * force(
                    x_0 - hx + (vals[k] + 1)*hx, shift_vals))
        return integral * hx * hy
    elif element == "phi_2_y":
        for k in range(deg):
            integral += wgts[k] * sum(
                wgts * [phi_2_y(x_0 - hx + (vals[k] + 1)*hx, y_val, x_0, y_0) for y_val in shift_vals] * force(
                    x_0 - hx + (vals[k] + 1)*hx, shift_vals))
        return integral * hx * hy
    else:
        for k in range(deg):
            integral += wgts[k] * sum(
                wgts * [phi_4(x_0 - hx + (vals[k] + 1)*hx, y_val, x_0, y_0) for y_val in shift_vals] * force(
                    x_0 - hx + (vals[k] + 1)*hx, shift_vals))
        return integral * hx * hy


# Main
# Setup
import numpy

# Setup A
mx = 17
m = mx * mx

A = numpy.zeros((m, m))

# A1
A1 = 256 / 45 * numpy.eye(mx) - 16 / 15 * numpy.eye(mx, mx, -1) - 16 / 15 * numpy.eye(mx, mx, 1)
for i in range(int(mx / 2)):
    A1[2 * i + 1, 2 * i + 1] = 176 / 45

# A2
A2 = -16 / 15 * numpy.eye(mx) - 16 / 45 * numpy.eye(mx, mx, -1) - 16 / 45 * numpy.eye(mx, mx, 1)
for i in range(int(mx / 2)):
    A2[2 * i + 1, 2 * i + 1] = -2 / 5
for i in range(int((mx - 2) / 2)):
    A2[2 * i + 1, 2 * i + 3] = 1 / 9
    A2[2 * i + 3, 2 * i + 1] = 1 / 9

# B1
B1 = 176 / 45 * numpy.eye(mx) - 2 / 5 * numpy.eye(mx, mx, -1) - 2 / 5 * numpy.eye(mx, mx, 1)
for i in range(int(mx / 2)):
    B1[2 * i + 1, 2 * i + 1] = 112 / 45
for i in range(int((mx - 2) / 2)):
    B1[2 * i + 1, 2 * i + 3] = -2 / 30
    B1[2 * i + 3, 2 * i + 1] = -2 / 30

# B2
B2 = A2

# B3
B3 = 1 / 9 * numpy.eye(mx, mx, -1) + 1 / 9 * numpy.eye(mx, mx, 1)
for i in range(int(mx / 2)):
    B3[2 * i + 1, 2 * i + 1] = - 2 / 30
for i in range(int((mx - 2) / 2)):
    B3[2 * i + 1, 2 * i + 3] = -1 / 45
    B3[2 * i + 3, 2 * i + 1] = -1 / 45

# Build A
A[:mx, :mx] = A1
for i in range(int(mx / 2)):
    A[mx * (2 * i + 1):mx * (2 * i + 2), mx * (2 * i + 1):mx * (2 * i + 2)] = B1
    A[mx * (2 * i + 2):mx * (2 * i + 3), mx * (2 * i + 2):mx * (2 * i + 3)] = A1
for i in range(int((mx - 1))):
    A[mx * i:mx * (i + 1), mx * (i + 1):mx * (i + 2)] = A2
    A[mx * (i + 1):mx * (i + 2), mx * i:mx * (i + 1)] = A2
for i in range(int((mx - 2) / 2)):
    A[mx * (2 * i + 1):mx * (2 * i + 2), mx * (2 * i + 3):mx * (2 * i + 4)] = B3
    A[mx * (2 * i + 3):mx * (2 * i + 4), mx * (2 * i + 1):mx * (2 * i + 2)] = B3

# Setup b
b = numpy.zeros(m)

for i in range(mx):
    for j in range(mx):
        if i % 2 == 0 and j % 2 == 0:
            # phi_1
            b[i * mx + j] = integrate("phi_1", (i + 1) / (mx + 1), (j + 1) / (mx + 1), 1 / (mx + 1), 1 / (mx + 1))
        elif i % 2 == 0:
            # phi_2_y
            b[i * mx + j] = integrate("phi_2_y", (i + 1) / (mx + 1), (j + 1) / (mx + 1), 1 / (mx + 1), 2 / (mx + 1))
        elif j % 2 == 0:
            # phi_2_x
            b[i * mx + j] = integrate("phi_2_x", (i + 1) / (mx + 1), (j + 1) / (mx + 1), 2 / (mx + 1), 1 / (mx + 1))
        else:
            # phi_4
            b[i * mx + j] = integrate("phi_4", (i + 1) / (mx + 1), (j + 1) / (mx + 1), 2 / (mx + 1), 2 / (mx + 1))

# Setup true u
u_true_vals = u_true(mx)

# Iterate to sove
u = fixed_pt(-A, b, numpy.zeros(m))
#u = numpy.linalg.inv(-A).dot(b)

print(u)
#print(u_true_vals)
print(max(u))
print(max(abs(u - u_true_vals)))
