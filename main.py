import numpy as np
import matplotlib.pyplot as plt

# Given parameters
a = 1
b = 1
t_interval = [a, a + 3]
target_accuracy = 1e-4

# Differential equation
def ode(t, y):
    return a * t + b * y

# Runge-Kutta-2 method
def runge_kutta_2(f, a, b, y0, h):
    t_values = [a]
    y_values = [y0]

    while t_values[-1] < b:
        t = t_values[-1]
        y = y_values[-1]
        k1 = h * f(t, y)
        k2 = h * f(t + h, y + k1)
        y_new = y + 0.5 * (k1 + k2)
        t_values.append(t + h)
        y_values.append(y_new)

    return t_values, y_values

# Milne-Simpson method with Euler for initial steps
def milne_simpson_euler(f, a, b, y0, h):

    t_euler, y_euler = [a], [y0]
    while t_euler[-1] < a + 2 * h:
        t = t_euler[-1]
        y = y_euler[-1]
        y_new = y + h * f(t, y)
        t_euler.append(t + h)
        y_euler.append(y_new)

    t_values = t_euler
    y_values = y_euler

    while t_values[-1] < b:
        t = t_values[-1]
        y = y_values[-1]

        # The predictive step of Euler's method
        t_pred = t + h
        y_pred = y + h * f(t, y)

        # The corrective step of the Milne-Simpson method
        y_corr = y + h / 3 * (f(t_pred, y_pred) + 4 * f(t, y) + f(t_values[-2], y_values[-2]))

        # Adding new time values ​​and functions to lists
        t_values.append(t_pred)
        y_values.append(y_corr)

    # We return the time and function values ​​at all steps
    return t_values, y_values


# Function to check accuracy
def check_accuracy(y1, y2):
    return np.max(np.abs(np.array(y1) - np.array(y2)))

# Initial values
h = 0.1  # Initial step size

# Get solutions
t_rk2, y_rk2 = runge_kutta_2(ode, *t_interval, b, h)
t_ms_euler, y_ms_euler = milne_simpson_euler(ode, *t_interval, b, h)

# Display numerical solutions
print("Runge-Kutta-2 Method:")
for t, y in zip(t_rk2, y_rk2):
    print(f"t = {t:.4f}, y = {y:.4f}")

print("\nMilne-Simpson with Euler Method:")
for t, y in zip(t_ms_euler, y_ms_euler):
    print(f"t = {t:.4f}, y = {y:.4f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t_rk2, y_rk2, label='Runge-Kutta-2')
plt.plot(t_ms_euler, y_ms_euler, label='Milne-Simpson with Euler')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution of the ODE')
plt.legend()
plt.grid(True)
plt.show()
