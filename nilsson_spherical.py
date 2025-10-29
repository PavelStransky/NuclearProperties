from matplotlib.lines import Line2D
import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from alive_progress import alive_bar

# -------------------------------------------------------------------------
# 1) Build 3D harmonic oscillator basis
# -------------------------------------------------------------------------
def build_basis(max_n=10, max_l=10, m=0):
    """Build spherical HO basis with quantum numbers n,l,m and energies."""
    N, L, M = [], [], []
    for l in range(abs(m), max_l + 1):
        for n in range(max_n + 1):
            N.append(n)
            L.append(l)
            M.append(m)
    return dict(N=np.array(N), L=np.array(L), M=np.array(M))

# -------------------------------------------------------------------------
# 2) Nilsson Hamiltonian with l^2 term
# -------------------------------------------------------------------------
def hamiltonian_nilsson_l2(index, delta, kappa=0.1):
    """
    Build Nilsson Hamiltonian: deformed HO + quadrupole + l^2 term.
    
    Parameters
    ----------
    index : dict with 'N','L','M','E'
    delta : float, deformation
    kappa : float, l^2 strength
    """
    N = index['N']
    L = index['L']
    M = index['M']
    dim = len(N)

    H = np.zeros((dim, dim), dtype=float)

    omega0 = np.sqrt((2 * np.exp(2 * delta) + np.exp(-4 * delta)) / 3)
    coefQ = (np.exp(2 * delta) - np.exp(-4 * delta)) / (6 * omega0)


    for i1 in range(dim):
        n1, l1, m1 = N[i1], L[i1], M[i1]
        for i2 in range(i1, dim):
            n2, l2, m2 = N[i2], L[i2], M[i2]

            if m1 != m2:
                continue  # only m1=m2 matrix elements

            # HO isotropic part
            p2 = 2 * n1 + l1 + 3/2 if i1 == i2 else 0.0

            # Quadrupole operator
            radial_term = 0.0
            spherical_term = 0.0    

            if l1 == l2:
                spherical_term = 2 * (l1 * (l1 + 1) - 3 * m1**2) / ((2 * l1 - 1) * (2 * l1 + 3))
            
                if n1 == n2:
                    radial_term = 2 * n1 + l1 + 3/2
                elif n2 == n1 + 1:
                    radial_term = -0.5 * np.sqrt((2*n1 + 2) * (2 * n1 + 2 * l1 + 3))
                elif n1 == n2 + 1:
                    radial_term = -0.5 * np.sqrt((2*n2 + 2) * (2 * n2 + 2 * l2 + 3))

            elif l2 == l1 + 2:
                spherical_term = 3 / (2 * l1 + 3) * np.sqrt((l1 + 1 + m1) * (l1 + 2 + m1) * (l1 + 1 - m1) * (l1 + 2 - m1) / ((2 * l1 + 1) * (2 * l1 + 5)))
                
                if n1 == n2:
                    radial_term = 0.5 * np.sqrt((2*n1 + 2*l1 + 3) * (2*n1 + 2*l1 + 5)) 
                elif n2 == n1 - 1:
                    radial_term = -np.sqrt(2*n1 * (2*n1 + 2*l1 + 3))
                elif n2 == n1 - 2:
                    radial_term = np.sqrt(n1 * (n1 - 1))

            elif l1 == l2 + 2:
                spherical_term = 3 / (2 * l2 + 3) * np.sqrt((l2 + 1 + m2) * (l2 + 2 + m2) * (l2 + 1 - m2) * (l2 + 2 - m2) / ((2 * l2 + 1) * (2 * l2 + 5)))
                
                if n1 == n2:
                    radial_term = 0.5 * np.sqrt((2*n2 + 2*l2 + 3) * (2*n2 + 2*l2 + 5))
                elif n1 == n2 - 1:
                    radial_term = -np.sqrt(2*n2 * (2*n2 + 2*l2 + 3))
                elif n1 == n2 - 2:
                    radial_term = np.sqrt(n2 * (n2 - 1))

            L2 = l1 * (l1 + 1) if i1 == i2 else 0.0
            averageN = 2 * n1 + l1 if i1 == i2 else 0.0

            H[i1, i2] = omega0 * p2 - coefQ * radial_term * spherical_term - kappa * (L2 - 0.5 * averageN * (averageN + 3))
            H[i2, i1] = H[i1, i2]

    return H

def plot_magic_numbers():
    magic_numbers = [10, 20, 35, 56, 84, 120, 165]
    for mn in magic_numbers:
        plt.axvline(mn, color='gray', linestyle='--', linewidth=0.8)
        plt.text(mn + 1, plt.ylim()[1] * 0.9, str(mn), rotation=90, color='gray')

# -------------------------------------------------------------------------
# 3) Diagonalize and plot
# -------------------------------------------------------------------------
def calculate_and_plot(kappa=0.02, max_m=20, max_n=30, max_l=25, num_delta=1001):
    nume = 3 * max_m

    deltas = np.linspace(-0.5, 0.5, num_delta)
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
                H = hamiltonian_nilsson_l2(basis, delta, kappa=kappa)
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
    plt.ylim(0, 10)
    plt.title("Spectrum of the Nilsson model + l^2 (no spin-orbit)\n(color by m)")
    plt.tight_layout()
    plt.show()

    delta_energies = np.array([np.sort(delta_energies[j,:]) for j in range(num_delta)])

    return delta_energies

fname = "nilsson_spectrum_spherical.dat"