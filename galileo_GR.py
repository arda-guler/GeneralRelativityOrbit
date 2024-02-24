# = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# GENERAL-RELATIVITY CORRECTED NUMERICAL ORBIT PROPAGATION
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Author: H. A. Guler
# Date: 2024-02-24
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# This work is based on the following paper:
#
# Sośnica, K., Bury, G., Zajdel, R. et al. General
# relativistic effects acting on the orbits of Galileo
# satellites. Celest Mech Dyn Astr 133, 14 (2021).
# https://doi.org/10.1007/s10569-021-10014-y
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Additional notes:
#
# This program ignores the de-Sitter component.
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

import numpy as np
import matplotlib.pyplot as plt

c = 299792458 # m s-1, time-space conversion factor
              # better known as "speed of light"

class Body:
    def __init__(self, mu, J=None):
        self.mu = mu
        self.J = J # angular momentum per mass

class Orbiter:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel

# I don't like typing 'np.' every time
def mag(x):
    return np.linalg.norm(x)

def cross(x, y):
    return np.cross(x, y)

def dot(x, y):
    return np.dot(x, y)

def get_grav_accel(body, orbiter):
    dist = mag(orbiter.pos)
    grav_dir = (-orbiter.pos) / dist
    grav_mag = body.mu / dist**2

    return grav_dir * grav_mag

def get_Schwarzchild(body, orbiter):
    global c

    beta = 1
    gamma = 1
    
    r = mag(orbiter.pos)
    accel = body.mu / (c**2 * r**3) * ((2 * (beta + gamma) * body.mu / r - gamma * dot(orbiter.vel, orbiter.vel)) * orbiter.pos + 2 * (1 + gamma) * dot(orbiter.pos, orbiter.vel) * orbiter.vel)

    return accel

def get_LenseThirring(body, orbiter):
    global c
    
    r = mag(orbiter.pos)
    gamma = 1

    accel = (1 + gamma) * body.mu / (c**2 * r**3)
    accel *= 3/r**2 * cross(orbiter.pos, orbiter.vel) * dot(orbiter.pos, body.J) + cross(orbiter.vel, body.J)

    return accel

# this is an 8th order symplectic integrator
def stepYoshida8(body, orbiter, dt, GR=True):
    # - - - CONSTANTS - - -
    w1 = 0.311790812418427e0
    w2 = -0.155946803821447e1
    w3 = -0.167896928259640e1
    w4 = 0.166335809963315e1
    w5 = -0.106458714789183e1
    w6 = 0.136934946416871e1
    w7 = 0.629030650210433e0
    w0 = 1.65899088454396 # (1 - 2 * (w1 + w2 + w3 + w4 + w5 + w6 + w7))

    ds = [w7, w6, w5, w4, w3, w2, w1, w0, w1, w2, w3, w4, w5, w6, w7]

    # cs = [w7 / 2, (w7 + w6) / 2, (w6 + w5) / 2, (w5 + w4) / 2,
    #           (w4 + w3) / 2, (w3 + w2) / 2, (w2 + w1) / 2, (w1 + w0) / 2,
    #           (w1 + w0) / 2, (w2 + w1) / 2, (w3 + w2) / 2, (w4 + w3) / 2,
    #           (w5 + w4) / 2, (w6 + w5) / 2, (w7 + w6) / 2, w7 / 2]

    cs = [0.3145153251052165, 0.9991900571895715, 0.15238115813844, 0.29938547587066, -0.007805591481624963,
          -1.619218660405435, -0.6238386128980216, 0.9853908484811935, 0.9853908484811935, -0.6238386128980216,
          -1.619218660405435, -0.007805591481624963, 0.29938547587066, 0.15238115813844, 0.9991900571895715,
          0.3145153251052165]

    for i in range(15):
        orbiter.pos = orbiter.pos + orbiter.vel * cs[i] * dt
        accel = get_grav_accel(body, orbiter)
        
        if GR: # Relativistic effects?
            accel_schwarzchild = get_Schwarzchild(body, orbiter)
            accel_lensethirring = get_LenseThirring(body, orbiter)
            accel = accel + accel_schwarzchild + accel_lensethirring
        else:
            accel_schwarzchild, accel_lensethirring = 0, 0
            
        orbiter.vel = orbiter.vel + accel * ds[i] * dt

    orbiter.pos = orbiter.pos + orbiter.vel * cs[15] * dt

    return accel_schwarzchild, accel_lensethirring

# this function was "stolen" from some helpful StackOverflow answer IIRC
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = 0
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = 0
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = 0

    plot_radius = 0.7*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_orbit(xs, ys, zs, sch, lts, v_schs, v_lts, GR):
    if GR:
        max_sch = max(sch)
        max_lts = max(lts)

        print("Max. Schwarzchild component (m s-2):", max_sch)
        print("Max. Lense-Thirring component (m s-2):", max_lts)

        # Schwarschild
        sch_colors = []
        for i in range(len(xs)):
            val = sch[i] / max_sch
            if val > 0.5:
                red = (val - 0.5) * 2
                green = 1 - red
                blue = 0

            else:
                blue = (0.5 - val) * 2
                green = 1 - blue
                red = 0

            sch_colors.append((red, green, blue))

        i = [arr[0] * 1e10 * 1e6 for arr in v_schs]
        j = [arr[1] * 1e10 * 1e6 for arr in v_schs]
        k = [arr[2] * 1e10 * 1e6 for arr in v_schs]

        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(xs, ys, zs, c=sch_colors)
        ax.quiver(xs, ys, zs, i, j, k, pivot='tail')

        # Plot Earth
        phi, theta = np.mgrid[0.0:2.0*np.pi:100j, 0.0:np.pi:50j]

        radius = 6371e3
        x = radius * np.sin(theta) * np.cos(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(theta)

        ax.plot_surface(x, y, z, color='b', alpha=0.6, edgecolors='k')
    
        set_axes_equal(ax)
        plt.title("Schwarzschild Component")

        # Lense-Thirring
        lts_colors = []
        for i in range(len(xs)):
            val = lts[i] / max_lts
            if val > 0.5:
                red = (val - 0.5) * 2
                green = 1 - red
                blue = 0

            else:
                blue = (0.5 - val) * 2
                green = 1 - blue
                red = 0

            lts_colors.append((red, green, blue))

        i = [arr[0] * 1e12 * 1e6 for arr in v_lts]
        j = [arr[1] * 1e12 * 1e6 for arr in v_lts]
        k = [arr[2] * 1e12 * 1e6 for arr in v_lts]

        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(xs, ys, zs, c=lts_colors)
        ax.quiver(xs, ys, zs, i, j, k, pivot='tail')

        ax.plot_surface(x, y, z, color='b', alpha=0.6, edgecolors='k')
        
        set_axes_equal(ax)
        plt.title("Lense-Thirring (Frame Dragging) Component")

    else:
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(xs, ys, zs)
        ax.set_box_aspect([1, 1, 1])

    plt.show()

def main():
    GM = 3.986004418e14 # Earth gravitational parameter
    J = np.array([0, 0, 7.05e33/5.972e24]) # kg m2 s-1
    earth = Body(GM, J)

    pos0 = np.array([20770e3, 0, 0])
    vel0 = np.array([0, -5e3, 1e3])
    satellite = Orbiter(pos0, vel0)

    enable_general_relativity = True

    end_time = 80000 # s
    dt = 64 # s
    N = end_time // dt
    xs = []
    ys = []
    zs = []
    schs = []
    lts = []
    v_schs = []
    v_lts = []
    for cycle in range(N):
        sch, lt = stepYoshida8(earth, satellite, dt, enable_general_relativity)
        xs.append(satellite.pos[0])
        ys.append(satellite.pos[1])
        zs.append(satellite.pos[2])
        schs.append(mag(sch))
        lts.append(mag(lt))
        v_schs.append(sch)
        v_lts.append(lt)

    plot_orbit(xs, ys, zs, schs, lts, v_schs, v_lts, enable_general_relativity)

main()

