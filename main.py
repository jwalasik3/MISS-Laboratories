# AI DISCLAIMER: AI has been used to suggest the libraries used in this code and explaied it's use cases and for stylistic improvements.
import csv
import numpy as np
import sympy as sp                # SymPy provides symbolic mathematics capabilities, allowing for algebraic manipulation, inserting variables and solving of equations (easy derivatives). Basically
import matplotlib.pyplot as plt
from scipy.optimize import brentq # Brentq is a root-finding algorithm that uses the Brent method, which is a mix of bisection and interpolation methods.

# Constants and configuration
SURFACE = "x**2 + y**2"         # Surface equation

P0 = (-5.0, 0.0, 200.0)         # initial object's position (x0, y0, z0)
V0 = (0.0, 0.0, 30.0)           # initial object's velocity  (v0x, v0y, v0z) [m/s]

G = 10.0                        # uniform gravitational acceleration [m/s^2]
MASS = 1.0                      # body mass [kg]
N_BOUNCES = 5                   # number of impacts to simulate

# Energy-loss coefficients k = E_kin(after) / E_kin(before). In other words, k = 1.0 means perfectly elastic collision, k = 0.0 means the body sticks to the surface.
K_VALUES = [1.0, 0.3]

# Plot settings
PLOT_RANGE = (-15, 15, -15, 15)  # Plot size
MESH_POINTS = 40                 # Mesh resolution
TRAJ_POINTS = 200                # Samples per trajectory
SAVE_FIGURES = False
SAVE_CSV = False
SHOW_FIGURES = True
OUTPUT_PREFIX = "test"          # Output files prefix
RESULTS_OUTPUT_DIR = "results"  # Output directory location

# Numerical settings for root finding
T_EPS = 1e-9                     # times <= T_EPS are treated as "zero" (rejected)
IMAG_TOL = 1e-9                  # roots with |imag| below this value are treated as real
T_MAX = 1e3                      # search horizon if the surface is not polynomial
T_SCAN_STEPS = 200000            # scan resolution for the non-polynomial fallback


# Symbolic expressions
x_s, y_s, t_s = sp.symbols("x y t", real=True)
f_expr = sp.sympify(SURFACE, locals={"x": x_s, "y": y_s}) # Conversion to sympify expression with specified variables x_s and y_s
fx_expr = sp.diff(f_expr, x_s)                            # df/dx
fy_expr = sp.diff(f_expr, y_s)                            # df/dy

f = sp.lambdify((x_s, y_s), f_expr, "numpy")
fx = sp.lambdify((x_s, y_s), fx_expr, "numpy")
fy = sp.lambdify((x_s, y_s), fy_expr, "numpy")


# Physics equations
def position(p0, v0, t):
    p0, v0 = np.asarray(p0, float), np.asarray(v0, float)
    t = np.asarray(t, float)
    return np.array([
        p0[0] + v0[0] * t,
        p0[1] + v0[1] * t,
        p0[2] + v0[2] * t - 0.5 * G * t ** 2,
    ])

# Velocity at time t in uniform gravitational field
def velocity(v0, t):
    return np.array([v0[0], v0[1], v0[2] - G * t])

# Find the time of impact for given position and velocity. Returns None if no impact is found.
def impact_time(p0, v0):
    x0, y0, z0 = p0
    vx, vy, vz = v0
    h = (z0 + vz * t_s - sp.Rational(1, 2) * sp.Float(G) * t_s ** 2) - f_expr.subs({x_s: x0 + vx * t_s, y_s: y0 + vy * t_s}, simultaneous=True)
    h = sp.expand(h)

    # Polynomial case
    try:
        poly = sp.Poly(h, t_s)                             # Express h as a polynomial surface in terms of time variable
        coeffs = [float(c) for c in poly.all_coeffs()]
        roots = np.roots(coeffs)
        real = roots[np.abs(roots.imag) < IMAG_TOL].real   # reject complex
        real = real[real > T_EPS]                          # reject <= 0
        return float(real.min()) if real.size else None
    except (sp.PolynomialError, TypeError, ValueError):
        pass

    # General (non-polynomial)
    hf = sp.lambdify(t_s, h, "numpy")
    ts = np.linspace(T_EPS * 10, T_MAX, T_SCAN_STEPS)
    vals = hf(ts)
    idx = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0] # Find sign changes - in continuous functions, a sign change indicates existance of root and allows to use Brentq method.
    if idx.size == 0:
        return None
    i = idx[0]
    return float(brentq(hf, ts[i], ts[i + 1]))


# Unit normal vector to the surface at (xi, yi)
def normal_vector(xi, yi):
    N = np.array([-fx(xi, yi), -fy(xi, yi), 1.0], dtype=float)
    return N / np.linalg.norm(N)


# Velocity after the bounce
def reflect(v_before, n, k):
    return np.sqrt(k) * (v_before - 2.0 * np.dot(v_before, n) * n)


# Simulation - returns a list of impacts (rows) and a list of trajectories for defined number of collisions (N_BOUNCES)
def simulate(k):
    p, v = np.array(P0, float), np.array(V0, float)
    rows, trajectories = [], []

    for _ in range(N_BOUNCES):
        tI = impact_time(p, v)
        if tI is None:
            print(f"No further collision found (after {len(rows)} impacts).")
            break

        # trajectory of this flight (for plotting)
        tt = np.linspace(0.0, tI, TRAJ_POINTS) # TRAJ_POINTS uniformly distributed time samples between 0 and collision time tI
        trajectories.append(position(p, v, tt))

        # impact point and incident velocity
        pI = position(p, v, tI)
        v_before = velocity(v, tI)

        # energies at the impact point (before the bounce)
        Ekin = 0.5 * MASS * np.dot(v_before, v_before)
        Epot = MASS * G * pI[2]
        rows.append(dict(xI=pI[0], yI=pI[1], zI=pI[2], dt=tI,
                         Ekin=Ekin, Epot=Epot, Etotal=Ekin + Epot))

        # new initial conditions for the next flight
        n = normal_vector(pI[0], pI[1])
        v = reflect(v_before, n, k)
        p = pI

    return rows, trajectories


# Output
COLUMNS = ["xI", "yI", "zI", "dt", "Ekin", "Epot", "Etotal"]
HEADERS = ["x_I", "y_I", "z_I", "dt", "E_kin", "E_pot", "E_total"]


def print_table(rows, k):
    print(f"\nResults (k = {k}):")
    print(" ".join(f"{h:>12}" for h in HEADERS))
    for r in rows:
        print(" ".join(f"{r[c]:12.4f}" for c in COLUMNS))


def save_csv(rows, filename):
    with open(RESULTS_OUTPUT_DIR + "/" + filename, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEADERS)
        for r in rows:
            w.writerow([f"{r[c]:.4f}" for c in COLUMNS])


# Used AI for plot generation -- AI GENERATED CONTENT
def plot_result(trajectories, k, filename=None):
    xmin, xmax, ymin, ymax = PLOT_RANGE
    X, Y = np.meshgrid(np.linspace(xmin, xmax, MESH_POINTS),
                       np.linspace(ymin, ymax, MESH_POINTS))
    Z = f(X, Y) * np.ones_like(X)          # ones_like handles constant surfaces

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, linewidth=0.4,
                      color="tab:blue", alpha=0.5)

    colors = plt.cm.autumn(np.linspace(0, 0.8, max(len(trajectories), 1)))
    for i, tr in enumerate(trajectories):
        ax.plot(tr[0], tr[1], tr[2], color=colors[i], linewidth=2.5,
                label=f"Trajectory {i + 1}")
        ax.scatter(tr[0][-1], tr[1][-1], tr[2][-1], color="k", s=25)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"Body bouncing off z = {SURFACE}  (k = {k})\n"
                 f"{len(trajectories)} trajectories, black dots = impact points")
    ax.legend(loc="upper left")
    fig.tight_layout()
    if filename:
        fig.savefig(RESULTS_OUTPUT_DIR + "/" + filename, dpi=200)
        print(f"  Figure saved: {filename}")
    return fig


def main():
    print(f"Surface: z = {SURFACE}")
    print(f"  df/dx = {fx_expr},  df/dy = {fy_expr}")
    print(f"P0 = {P0},  V0 = {V0},  g = {G},  m = {MASS}")

    for k in K_VALUES:
        rows, trajectories = simulate(k)
        print_table(rows, k)
        tag = f"{OUTPUT_PREFIX}_k{k}"
        if SAVE_CSV:
            save_csv(rows, f"{tag}.csv")
            print(f"  Table saved: {tag}.csv")
        plot_result(trajectories, k, f"{tag}.png" if SAVE_FIGURES else None)

    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()
