# MOID Solver
# AE 498 Planetary Defense

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root
from mpl_toolkits.mplot3d import Axes3D  
earth_kep   = (1.4765067E+08, 9.1669995E-03, 4.2422693E-03, 6.64375167E+01, 1.4760836E+01)  # (a,e,i,raan,argp)
apophis_kep = (137986131, 0.1911663355386932, 3.340958441017069, 126.6728325163065, 203.8996515621043)
YR4_kep = (3.7680703E+08, 6.6164147E-01, 3.4001497E+00, 1.3429905E+02, 2.7147904E+02)
ATLAS_kep = (-3.9552667E+07, 6.1469268E+00, 1.7512507E+02, 1.2817255E+02, 3.2228906E+02)
AU_km = 1.495978707e8
def kep2cart(a, e, i, raan, argp, nu):
    i = np.radians(i)
    raan = np.radians(raan)
    argp = np.radians(argp)
    nu = np.radians(nu)

    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu))

    x_p = r * np.cos(nu)
    y_p = r * np.sin(nu)

    R = np.array([
        [np.cos(raan) * np.cos(argp) - np.sin(raan) * np.sin(argp) * np.cos(i),
         -np.cos(raan) * np.sin(argp) - np.sin(raan) * np.cos(argp) * np.cos(i),
         np.sin(raan) * np.sin(i)],
        [np.sin(raan) * np.cos(argp) + np.cos(raan) * np.sin(argp) * np.cos(i),
         -np.sin(raan) * np.sin(argp) + np.cos(raan) * np.cos(argp) * np.cos(i),
         -np.cos(raan) * np.sin(i)],
        [np.sin(argp) * np.sin(i),
         np.cos(argp) * np.sin(i),
         np.cos(i)]
    ])

    return R @ np.array([x_p, y_p, 0.0])
def kep2cart_hyperbolic_H(a, e, i, raan, argp, H):
    if e <= 1.0:
        raise ValueError("Hyperbolic case requires e > 1.")
    if a >= 0.0:
        raise ValueError("For a hyperbola, a is typically negative. (Use a < 0.)")

    i = np.radians(i)
    raan = np.radians(raan)
    argp = np.radians(argp)

    
    x_p = a * (np.cosh(H) - e)
    y_p = a * np.sqrt(e**2 - 1.0) * np.sinh(H)

    R = np.array([
        [np.cos(raan) * np.cos(argp) - np.sin(raan) * np.sin(argp) * np.cos(i),
         -np.cos(raan) * np.sin(argp) - np.sin(raan) * np.cos(argp) * np.cos(i),
         np.sin(raan) * np.sin(i)],
        [np.sin(raan) * np.cos(argp) + np.cos(raan) * np.sin(argp) * np.cos(i),
         -np.sin(raan) * np.sin(argp) + np.cos(raan) * np.cos(argp) * np.cos(i),
         -np.cos(raan) * np.sin(i)],
        [np.sin(argp) * np.sin(i),
         np.cos(argp) * np.sin(i),
         np.cos(i)]
    ])

    return R @ np.array([x_p, y_p, 0.0])

def MOID_grid_return_argmin(r1, r2, fs1, fs2):
    diff = r1[:, None, :] - r2[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    i1, i2 = np.unravel_index(np.argmin(d2), d2.shape)

    return float(fs1[i1]), float(fs2[i2]), float(d2[i1, i2]), d2


def refine_min(r1_of, r2_of, f10, f20, h=1e-4):
    # h in degrees 
    def dr_df(r_of, f):
        return (r_of(f + h) - r_of(f - h)) / (2*h)

    def F(x):
        f1, f2 = x
        r1 = r1_of(f1)
        r2 = r2_of(f2)
        d  = r1 - r2
        t1 = dr_df(r1_of, f1)
        t2 = dr_df(r2_of, f2)
        return np.array([d @ t1, -(d @ t2)])  

    sol = root(F, x0=np.array([f10, f20]), method="hybr")
    f1, f2 = sol.x
    d = r1_of(f1) - r2_of(f2)
    return f1, f2, np.linalg.norm(d), sol.success, sol.message

def MOID_plot(r1_of, r2_of, fs1, fs2, f10, f20, d2_grid,
              Orbit_1_name="Orbit 1", Orbit_2_name="Orbit 2"):

    f1_star, f2_star, _, _, _ = refine_min(r1_of, r2_of, f10, f20)

    F1, F2 = np.meshgrid(fs1, fs2, indexing="ij")

    plt.figure(figsize=(8, 5.5))
    levels = 60
    plt.contour(F1, F2, d2_grid, levels=levels, linewidths=0.7)
    plt.contourf(F1, F2, d2_grid, levels=levels, alpha=0.85)
    plt.plot(f1_star, f2_star, "r+", markersize=12, markeredgewidth=2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(rf"{Orbit_1_name} $f_1$ (deg)", fontsize= 18)
    plt.ylabel(rf"{Orbit_2_name} $f_2$ (deg)", fontsize = 18)
    #plt.title(rf"$d^2(f_1,f_2)$ (km$^2$) between {Orbit_1_name} and {Orbit_2_name}")
    plt.tight_layout()
    plt.show()

###
Plot = True
fs1 = np.arange(0.0, 360.0, 0.1)  
N = fs1.size
e2 = ATLAS_kep[1]
nu_inf = np.degrees(np.arccos(-1.0 / e2))  # asymptote true anomaly (deg)
margin = 0.01  
lfs1 = len(fs1)
fs2 = np.linspace(-nu_inf + margin, nu_inf - margin, lfs1)
Earth_r   = np.array([kep2cart(*earth_kep,   f) for f in fs1])   # (N,3)
Apophis_r = np.array([kep2cart(*apophis_kep, f) for f in fs1])   # (N,3)
yr4_r = np.array([kep2cart(*YR4_kep, f) for f in fs1])   # (N,3)
ATLAS_r = np.array([kep2cart_hyperbolic_H(*ATLAS_kep, np.radians(f)) for f in fs2])  


cases = [
    ("Earth", "Apophis",
     lambda f: kep2cart(*earth_kep, f),
     lambda f: kep2cart(*apophis_kep, f),
     Earth_r, Apophis_r, fs1, fs1),

    ("Earth", "2024 YR4",
     lambda f: kep2cart(*earth_kep, f),
     lambda f: kep2cart(*YR4_kep, f),
     Earth_r, yr4_r, fs1, fs1),

    ("Apophis", "2024 YR4",
     lambda f: kep2cart(*apophis_kep, f),
     lambda f: kep2cart(*YR4_kep, f),
     Apophis_r, yr4_r, fs1, fs1),

    ("Earth", "3I/ATLAS",
     lambda f: kep2cart(*earth_kep, f),
     lambda f: kep2cart_hyperbolic_H(*ATLAS_kep, np.radians(f)),
     Earth_r, ATLAS_r, fs1, fs2),
]
print("------------------------------------------------------------------------------------------------------")
print(f"{'Orbit 1':<12} {'Orbit 2':<12} {'MOID [AU]':<18} {'MOID [KM]':<18} {'f1 (deg)':<18} {'f2 (deg)':<18}")
print("------------------------------------------------------------------------------------------------------")
for name1, name2, r1_of, r2_of, R1, R2, fgrid1, fgrid2 in cases:
    # --- Coarse grid ---
    f10, f20, d2min, d2 = MOID_grid_return_argmin(R1, R2, fgrid1, fgrid2)
    # ---  Refine minimum ---
    f1_zoom, f2_zoom, moid_zoom, ok, msg = refine_min(
        r1_of, r2_of,
        f10, f20,
        h=1e-3
    )
    print(f"{name1:<12} {name2:<12} {moid_zoom/AU_km:<18.10f} {moid_zoom:<18.6f} {f1_zoom:<18.9f} {f2_zoom:<18.9f}")

if Plot == True:
    for name1, name2, r1_of, r2_of, R1, R2, fgrid1, fgrid2 in cases:
        f10, f20, d2min, d2grid = MOID_grid_return_argmin(R1, R2, fgrid1, fgrid2)
        MOID_plot(r1_of, r2_of, fgrid1, fgrid2, f10, f20, d2grid,
                  Orbit_1_name=name1, Orbit_2_name=name2)
