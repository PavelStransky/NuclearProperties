import numpy as np
from scipy.linalg import eigh
from scipy.special import spherical_jn
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from alive_progress import alive_bar

# -------------------------------------------------------------------------
# 1) Compute zeros of spherical Bessel function jl(x)
# -------------------------------------------------------------------------
def spherical_bessel_zeros(l, max_n, max_x=200):
    """
    Return first max_n zeros of spherical Bessel function j_ln(x)
    using sign-change detection for robust root bracketing.
    """
    zeros = []
    x = 0.01
    dx = 0.01
    f_prev = spherical_jn(l, x)
    
    while len(zeros) < max_n and x < max_x:
        x += dx
        f = spherical_jn(l, x)
        if f_prev * f < 0:
            # root is bracketed between x-dx and x
            root = brentq(lambda t: spherical_jn(l, t), x - dx, x)
            zeros.append(root)
        f_prev = f
    
    return np.array(zeros)

# -------------------------------------------------------------------------
# 2) Build basis
# -------------------------------------------------------------------------
def build_basis(max_n=10, max_l=10, m=0):
    """Generate index dict with all (n,l,m) up to given lmax, nmax"""
    N, L, M, E = [], [], [], []
    for l in range(np.abs(m), max_l + 1):
        zeros = spherical_bessel_zeros(l, max_n)
        for n, z in enumerate(zeros):
            N.append(n + 1)
            L.append(l)
            M.append(m)
            E.append(z**2)

    return dict(N=np.array(N), L=np.array(L), M=np.array(M), E=np.array(E))


# -------------------------------------------------------------------------
# 3) Build Hamiltonian
# -------------------------------------------------------------------------
def hamiltonian_ellipsoid(index, delta):
    """Build Hamiltonian matrix for ellipsoidal infinite well."""
    N = index['N']
    L = index['L']
    M = index['M']
    E = index['E']
    dim = len(N)

    koef1 = (2.0 * np.exp(2.0 * delta) + np.exp(-4.0 * delta)) / 3.0
    koef2 = -2.0 * (np.exp(2.0 * delta) - np.exp(-4.0 * delta)) / 3.0

    H = np.zeros((dim, dim), dtype=float)

    for i1 in range(dim):
        n1, l1, m1, e1 = N[i1], L[i1], M[i1], E[i1]
        for i2 in range(i1, dim):
            n2, l2, m2, e2 = N[i2], L[i2], M[i2], E[i2]

            if m1 != m2:
                continue  # only m1=m2 matrix elements

            p2 = e1 if i1 == i2 else 0.0
            T2 = 0.0

            if l1 == l2 and n1 == n2:
                T2 = e1 * (l1 * (l1 + 1) - 3 * m1 * m1) / ((2 * l1 - 1) * (2 * l1 + 3))
            elif l2 == l1 + 2:
                num = (l1 + 1 + m1) * (l1 + 2 + m1) * (l1 + 1 - m1) * (l1 + 2 - m1)
                den = (2 * l1 + 1) * (2 * l1 + 5)
                T2 = np.sqrt(e1 * e2) / (e2 - e1) * 3.0 * np.sqrt(num / den)
            elif l1 == l2 + 2:
                num = (l2 + 1 + m2) * (l2 + 2 + m2) * (l2 + 1 - m2) * (l2 + 2 - m2)
                den = (2 * l2 + 1) * (2 * l2 + 5)
                T2 = np.sqrt(e1 * e2) / (e1 - e2) * 3.0 * np.sqrt(num / den)

            H[i1, i2] = koef1 * p2 + koef2 * T2
            H[i2, i1] = H[i1, i2]

    return H

# -------------------------------------------------------------------------
# 4) Main: Diagonalize and plot
# -------------------------------------------------------------------------
def calculate_and_plot(max_m=15, max_n=25, max_l=25, num_delta=1001):
    nume = 3 * max_m

    deltas = np.linspace(-0.25, 0.25, num_delta)
    delta_energies = np.zeros((num_delta, nume * (2 * max_m + 1) + 1))
    delta_energies[:,0] = deltas

    ms = list(range(max_m + 1))

    cmap = plt.get_cmap("tab20", max_m + 1)
    color_for_m = {m: cmap(i) for i, m in enumerate(ms)}

    for m in ms:
        basis = build_basis(max_n=max_n, max_l=max_l, m=m)
        energies = []

        with alive_bar(num_delta, title=f"Computing m={m}...") as bar:
            for delta in deltas:
                H = hamiltonian_ellipsoid(basis, delta)
                evals, evecs = eigh(H)
                energies.append(evals[:nume])
                bar()

        energies = np.array(energies)
        delta_energies[:, (max_m + m) * nume + 1:(max_m + m + 1) * nume + 1] = energies
        delta_energies[:, (max_m - m) * nume + 1:(max_m - m + 1) * nume + 1] = energies

        plt.plot(deltas, energies, color=color_for_m[m], linewidth=1.5)

    legend_lines = [Line2D([0],[0], color=color_for_m[m], lw=2, label=f"|m|={m}") for m in ms]
    plt.legend(handles=legend_lines, loc='upper right', fontsize=9)
    plt.xlabel("Deformation δ")
    plt.ylabel("Energy (a.u.)")
    plt.ylim(0, 160)
    plt.title("Spectrum of 3D Ellipsoidal Infinite Well\n(color by m, line style by parity)")
    plt.tight_layout()
    plt.show()

    delta_energies = np.array([np.sort(delta_energies[j,:]) for j in range(num_delta)])

    return delta_energies

def plot_magic_numbers():
    magic_numbers = [10, 29, 46, 69, 93, 127, 169]
    for mn in magic_numbers:
        plt.axvline(mn, color='gray', linestyle='--', linewidth=0.8)
        plt.text(mn + 1, plt.ylim()[1] * 0.9, str(mn), rotation=90, color='gray')

fname = "ellipsoid_spectrum.dat"