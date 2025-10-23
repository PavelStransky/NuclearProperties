import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit

rcParams["figure.figsize"] = (6, 4.7)

""" All the numbers are in MeVs """

# Constants (masses of proton, neutron, electron and atomic constant)
Mp = 938.2720882
Mn = 939.5654205
Me = 0.510998950
u = 931.4941024

def import_masses():
    """ Reads binding energies and masses from the AME2020 data file """
    nuclides = []

    with open(r"mass_1.mas20.txt") as file:
        # We skip the header
        for _ in range(37):
            file.readline()

        for line in file:
            nuclide = {}

            nuclide["N"] = int(line[4:9])
            nuclide["Z"] = int(line[10:14])
            nuclide["A"] = int(line[15:19])

            # Check if A = N + Z
            if nuclide["N"] + nuclide["Z"] != nuclide["A"]:
                print(f"Inconsistency at N={nuclide['N']}, Z={nuclide['Z']}")

            nuclide["symbol"] = line[20:22].strip()

            # Binding energy per nucleon (small b)
            b = line[54:66]

            # We skip nonexperimental values marked by the # sign
            if b.find('#') >= 0:
                continue

            # We convert keVs into MeVs
            nuclide["b"] = float(b) / 1000
            nuclide["berror"] = float(line[68:77]) / 1000

            # Binding energy (capital B)
            nuclide["B"] = nuclide["b"] * nuclide["A"]
            nuclide["Berror"] = nuclide["berror"] * nuclide["A"]

            # We calculate the nuclear mass (can be also used the last column from the data file)
            nuclide["M"] = Mn * nuclide["N"] + Mp * nuclide["Z"] - nuclide["B"]

            nuclides.append(nuclide)

    return nuclides


def import_spins(nuclides):
    ignore_characters = ['#', '(', ')', ' ', '*']

    """ Reads spins from the data file and adds them to the nuclides array """
    with open(r"nubase_4.mas20.txt") as file:
        # We skip the header
        for _ in range(25):
            file.readline()

        for line in file:
            A = int(line[0:3])
            Z = int(line[4:7])
            iso = int(line[7:8])
            
            spin_str = line[88:96]

            # We skip nonexperimental values marked by the # sign
            if spin_str.find('#') >= 0:
                continue

            # We skip if more values are given
            if spin_str.find(',') >= 0:
                continue

            for char in ignore_characters:
                spin_str = spin_str.replace(char, "")

            if spin_str == "":
                continue

            parity = 0
            spin = 0

            if spin_str[-1] == '+':
                parity = 1
                spin_str = spin_str.replace('+', '')
            elif spin_str[-1] == '-':
                parity = -1
                spin_str = spin_str.replace('-', '')                

            try:
                fraction = spin_str.split('/')
                if len(fraction) == 2:
                    spin = int(fraction[0])
                else:
                    spin = 2 * int(fraction[0])
            except ValueError:
                continue

            # print(A, Z, iso, spin_str, spin, parity)

            for nuclide in nuclides:
                if nuclide["A"] == A and nuclide["Z"] == Z:
                    if "isomer" in nuclide and nuclide["isomer"] < iso:
                        continue

                    nuclide["spin"] = spin
                    nuclide["parity"] = parity
                    nuclide["isomer"] = iso
                    break

    for nuclide in nuclides:
        if "spin" in nuclide:
                continue

        nuclide["spin"] = 0
        nuclide["parity"] = 0
        nuclide["isomer"] = 0

    return

def max_nza(nuclides):
    """ Returns maximum values of the neutron number, proton number and mass number """
    max_N = 0
    max_Z = 0
    max_A = 0

    for nuclide in nuclides:
        max_N = max(nuclide["N"], max_N)
        max_Z = max(nuclide["Z"], max_Z)
        max_A = max(nuclide["A"], max_A)

    return max_N, max_Z, max_A


def sort_nuclides(nuclides, name="b"):
    """ Sorts the nuclides array according to the given criterion (Python dictionaries are ordered since Python 3.7) """
    keys = [nuclide[name] for nuclide in nuclides]
    return [nuclide for _, nuclide in sorted(zip(keys, nuclides))]


def create_matrix(nuclides, name="b"):
    """ Returns the element with the given name as a 2D array """
    max_N, max_Z, _ = max_nza(nuclides)

    result = np.zeros((max_N + 1, max_Z + 1))

    for nuclide in nuclides:
        result[nuclide["N"], nuclide["Z"]] = nuclide[name]

    return result


def colormap(elements=2048):
    """ An auxiliary function to change the colormap - large values in black """
    cmap = colormaps.get_cmap("jet")
    cmap = cmap(np.linspace(0, 1, elements))
    # cmap[:1, :] = [1, 1, 1, 1]        # Zero values (usually not measured values)
    cmap[-1:, :] = [0, 0, 0, 1]       # Large values
    cmap = ListedColormap(cmap)

    return cmap


def plot_heatmap(data_matrix, title="", vmin=None, vmax=None, show_colorbar=True, label='B/A [MeV]', elements=2048):
    """ Plots a heatmap of the given data matrix """
    # # Mask zeros
    data_masked = np.ma.masked_where(data_matrix == 0, data_matrix)

    # # Create colormap that leaves masked values as white
    cmap = colormap(elements)
    cmap.set_bad(color='white')  # masked values appear white

    if not show_colorbar and vmin is None:
        vmin = 0

    # plt.pcolormesh(range(data_matrix.shape[0]), range(data_matrix.shape[1]), np.transpose(data_masked), shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.imshow(np.transpose(data_masked), origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')

    # Plot magic numbers
    magics = [2, 8, 20, 28, 50, 82, 126]
    for magic in magics:
        plt.axvline(x=magic, color='gray', alpha=0.5, linestyle='--', linewidth=0.8)
        plt.axhline(y=magic, color='gray', alpha=0.5, linestyle='--', linewidth=0.8)

    if show_colorbar:
        plt.colorbar(label=label)

    plt.title(title)

    plt.xlim(0, data_matrix.shape[0])
    plt.ylim(0, data_matrix.shape[1])

    plt.xlabel('N')
    plt.ylabel('Z')
    plt.show()


def Bethe_Weizsacker(N, Z, a_v=15.76, a_s=17.81, a_c=0.711, a_a=23.7, delta=34, delta_exp=3/4):
    """ Calculate binding energy in MeV using the Bethe-Weizsäcker formula.
        N : int or array-like
        Z : int or array-like
        a_v, a_s, a_c, a_a, delta : coefficients in MeV 
        delta_exp : exponent for pairing term (default 3/4)
    """
    N = np.array(N)
    Z = np.array(Z)

    A = N + Z

    result = np.array(a_v * A
            - a_s * A**(2/3)
            - a_c * Z * (Z - 1) / A**(1/3)
            - a_a * (N - Z)**2 / A)

    odd_mask = (N % 2 == 1) & (Z % 2 == 1)
    even_mask = (N % 2 == 0) & (Z % 2 == 0)

    result[odd_mask] -= delta / (A[odd_mask] ** (delta_exp))
    result[even_mask] += delta / (A[even_mask] ** (delta_exp))

    return result


def most_bound_nuclei(nuclides, num=10):
    """ Print the most bound nuclei according to binding energy per nucleon """
    nuclides_sorted = sort_nuclides(nuclides, name="b")[::-1]

    print(f"Top {num} most bound nuclei:")
    for i in range(1, num + 1):
        print(f"{i}: {nuclides_sorted[i]['symbol']} Z={nuclides_sorted[i]['Z']}, N={nuclides_sorted[i]['N']}, BA={nuclides_sorted[i]['b']:.4f} MeV")


def separation(nuclides, N_diff=0, Z_diff=0):
    """ Calculate separation energy matrix
        N_diff : difference in neutron number (e.g., 2 for 2n separation energy)
        Z_diff : difference in proton number (e.g., 2 for 2p separation energy)
    """
    M_matrix = create_matrix(nuclides, name="M")
    result = np.zeros_like(M_matrix)

    M_diff = Z_diff * Mp + N_diff * Mn

    if Z_diff != 0 and N_diff != 0 and M_matrix[N_diff, Z_diff] != 0:
        M_diff = M_matrix[N_diff, Z_diff]  # If we separate an existing nucleus, we use its mass

    for N in range(N_diff, M_matrix.shape[0]):
        for Z in range(Z_diff, M_matrix.shape[1]):
            if M_matrix[N, Z] != 0 and M_matrix[N - N_diff, Z - Z_diff] != 0:
                result[N, Z] = M_matrix[N - N_diff, Z - Z_diff] - M_matrix[N, Z] + M_diff

    return result


def plot_beta_lines(matrix, As=range(30, 40), title=""):
    """ Plot Beta lines (A = const) for given A range """
    Zs = np.array(range(matrix.shape[1]), dtype=int)

    for A in As:
        Nx = -Zs + A
        mask = Nx > 0

        Nx = Nx[mask]
        if len(Nx) == 0:
            continue

        Zx = Zs[mask]

        line = matrix[Nx, Zx]

        if np.count_nonzero(line) == 0:
            continue

        mask = line != 0

        Zx = Zx[mask]
        line = line[mask]

        plt.plot(Zx, line, 'o-', label=f'A={A}')

    plt.title(title)
    plt.xlabel('Z')
    plt.ylabel('B/A [MeV]')
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.show()


def plot_isotope_lines(nuclides, matrix, Zs=None, title="", label="B/A [MeV]"):
    """ Plot isotope lines for given Z range """

    if Zs is None:
        Zs = range(matrix.shape[1])

    for Z in Zs:
        line = matrix[:, Z]

        if np.count_nonzero(line) == 0:
            continue

        Ns = np.array(range(matrix.shape[0]))

        mask = line != 0

        Ns = Ns[mask]
        line = line[mask]

        symbols = [nuclide['symbol'] for nuclide in nuclides if nuclide['Z'] == Z]
        symbol = symbols[0] if symbols else ""

        plt.plot(Ns, line, 'o-', alpha=0.7, label=f'{Z}{symbol}')

    plt.title(title)
    plt.xlabel('N')
    plt.ylabel(label)
    plt.legend(fontsize='small', ncol=len(Zs) // 30 + 1)
    plt.tight_layout()
    plt.show()


def plot_iso_lines(nuclides, matrix, Z=20, N=20, title="", label="J"):
    """ Plot both isotope and isotone lines for a given N, Z nucleus """

    line = matrix[:, Z]
    Ns = np.array(range(matrix.shape[0]))

    mask = line != 0

    Ns = Ns[mask]
    line = line[mask]

    plt.plot(Ns, line, 'o-', alpha=0.7, label=f'isotopes Z={Z}')


    line = matrix[N, :]
    Zs = np.array(range(matrix.shape[1]))

    mask = line != 0

    Zs = Zs[mask]
    line = line[mask]

    plt.plot(Zs, line, 'o-', alpha=0.7, label=f'isotones N={N}')

    plt.title(title)
    plt.xlabel('Z,N')
    plt.ylabel(label)
    plt.legend()
    plt.show()


def compare_fits(nuclides):
    Ns = np.array([nuclide['N'] for nuclide in nuclides])
    Zs = np.array([nuclide['Z'] for nuclide in nuclides])

    NZs = np.vstack((Ns, Zs))
    As = Ns + Zs

    bs = np.array([nuclide['b'] for nuclide in nuclides])

    length = len(bs)

    def BW_fit_B(NZ, a_v=15.76, a_s=17.81, a_c=0.711, a_a=23.7, delta=33.5):
        """ Wrapper for Bethe-Weizsäcker to fit B """
        N, Z = NZ
        return Bethe_Weizsacker(N, Z, a_v, a_s, a_c, a_a, delta)

    def BW_fit_b(NZ, a_v=15.76, a_s=17.81, a_c=0.711, a_a=23.7, delta=33.5):
        """ Wrapper for Bethe-Weizsäcker to fit the binding energy per nucleon """
        N, Z = NZ
        return Bethe_Weizsacker(N, Z, a_v, a_s, a_c, a_a, delta) / (N + Z)

    def BW_fit_B_noAC(NZ, a_v=15.76, a_s=17.81, a_a=23.7, delta=33.5):
        """ Wrapper for Bethe-Weizsäcker to fit B without Coulomb term """
        N, Z = NZ
        return Bethe_Weizsacker(N, Z, a_v, a_s, 0.711, a_a, delta)

    def BW_fit_B_pairing12(NZ, a_v=15.76, a_s=17.81, a_c=0.711, a_a=23.7, delta=33.5):
        """ Wrapper for Bethe-Weizsäcker to fit B with pairing term proportional to A^(-1/2) """
        N, Z = NZ
        return Bethe_Weizsacker(N, Z, a_v, a_s, a_c, a_a, delta, delta_exp=1/2)

    def fit(params, fit_function, per_nucleon=False):
        p_values, covariance = curve_fit(fit_function, NZs, bs * As if not per_nucleon else bs, p0=list(params.values()))
        res = bs * As - fit_function(NZs, *p_values) * (1 if not per_nucleon else As)

        avg_res = np.sqrt(np.sum(res**2)) / length
        print(f"{fit_function.__name__} (R={avg_res:.3f})", [f"{name}={value:.3f}+{error:.3f}" for name, value, error in zip(params.keys(), p_values, np.sqrt(np.diag(covariance)))])

        return p_values

    res = bs - Bethe_Weizsacker(Ns, Zs)
    print(f"Wiki (LS2) (R={np.sqrt(np.sum(res**2)) / length:.3f})")

    params = {"a_v": 15.75, "a_s": 17.8, "a_c": 0.711, "a_a": 23.7, "delta": 34}
    fit(params, BW_fit_b, per_nucleon=True)

    params = {"a_v": 15.75, "a_s": 17.8, "a_c": 0.711, "a_a": 23.7, "delta": 34}
    fit(params, BW_fit_B, per_nucleon=False)

    params = {"a_v": 15.75, "a_s": 17.8, "a_a": 23.7, "delta": 34}
    fit(params, BW_fit_B_noAC, per_nucleon=False)

    params = {"a_v": 15.75, "a_s": 17.8, "a_c": 0.711, "a_a": 23.7, "delta": 34}
    return fit(params, BW_fit_B_pairing12, per_nucleon=False)



def Bethe_Weizsacker_deviation(nuclides, *params):
    """ Calculate deviation matrix from Bethe-Weizsäcker formula. """
    max_N, max_Z, _ = max_nza(nuclides)

    difference_matrix = np.zeros((max_N + 1, max_Z + 1))

    for nuclide in nuclides:
        N = nuclide['N']
        Z = nuclide['Z']
        B = nuclide['B']
        difference_matrix[N, Z] = B - Bethe_Weizsacker(N, Z, *params)

    return difference_matrix


def neutron_drip_line(*params):
    """ Neutron drip line """    
    Ns = range(1, 160)

    drip_line = []
    Zs = []

    for Z in range(1, 120):
        B = Bethe_Weizsacker(Ns, Z, *params)
        max_i = np.argmax(B)
        if max_i < len(B) - 1:
            Zs.append(Z)
            drip_line.append(Ns[max_i])

    return np.array(drip_line, dtype=int), np.array(Zs, dtype=int)


def proton_drip_line(*params):
    """ Proton drip line """    
    Zs = range(1, 120)

    drip_line = []
    Ns = []

    for N in range(1, 200):
        B = Bethe_Weizsacker(N, Zs, *params)
        max_i = np.argmax(B)
        if max_i < len(B) - 1:
            Ns.append(N)
            drip_line.append(Zs[max_i])

    return np.array(Ns, dtype=int), np.array(drip_line, dtype=int)


def valley_of_stability(*params):
    """ Valley of Stability """
    Zs = np.array(range(1, 120), dtype=int)

    drip_line = []
    Ns = []

    for A in range(1, 300):
        Nx = -Zs + A
        mask = Nx > 0

        Nx = Nx[mask]
        if len(Nx) == 0:
            continue

        B = Bethe_Weizsacker(Nx, Zs[mask], *params)
        max_i = np.argmax(B)
        if max_i < len(B) - 1:
            Ns.append(Nx[max_i])
            drip_line.append(Zs[mask][max_i])

    return np.array(Ns, dtype=int), np.array(drip_line, dtype=int)


def plot_drip_lines(*best_params):
    plt.plot(*neutron_drip_line(*best_params), color='blue', linestyle='-', linewidth=1, label='Neutron Drip Line')
    plt.plot(*proton_drip_line(*best_params), color='green', linestyle='-', linewidth=1, label='Proton Drip Line')
    plt.plot(*valley_of_stability(*best_params), color='black', linestyle='-', linewidth=1, label='Valley of Stability')
    plt.legend()


def alpha_decay(nuclides):
    """ Finds all possible nuclides that can decay via alpha process """

    M_matrix = create_matrix(nuclides, name="M")

    # Hellium - alpha particle
    M_alpha = M_matrix[2, 2]

    alpha = np.zeros_like(M_matrix, dtype=int)

    max_Z, max_N = M_matrix.shape

    for Z in range(max_Z):
        for N in range(max_N):
            if N <= 2 or Z <= 2 or M_matrix[Z, N] == 0 or M_matrix[Z - 2, N - 2] == 0:
                continue

            mass_parent = M_matrix[Z, N]
            mass_daughter = M_matrix[Z - 2, N - 2]

            if mass_parent > mass_daughter + M_alpha:
                alpha[Z, N] = 1
            else:
                alpha[Z, N] = 2

    return alpha


def beta_decay(nuclides):
    """ Finds all possible nuclides that can decay via beta process """

    M_matrix = create_matrix(nuclides, name="M")

    beta = np.zeros_like(M_matrix, dtype=int)

    max_Z, max_N = M_matrix.shape

    for Z in range(max_Z):
        for N in range(max_N):
            if M_matrix[Z, N] == 0:
                continue

            if Z >= max_Z - 1 or M_matrix[Z + 1, N - 1] == 0:
                continue

            if N >= max_N - 1 or M_matrix[Z - 1, N + 1] == 0:
                continue

            mass_parent = M_matrix[Z, N]
            mass_betaminus = M_matrix[Z + 1, N - 1]
            mass_betaplus = M_matrix[Z - 1, N + 1]

            if N > 1 and mass_parent > mass_betaminus + Mn - Mp:
                beta[Z, N] = 1
            elif Z > 1 and mass_parent > mass_betaplus - Mn + Mp:
                beta[Z, N] = 2
            else:
                beta[Z, N] = 3

    return beta


def plot_nuclei_along_valley_of_stability(nuclides, params):
    Ns_valley, Zs_valley = valley_of_stability(*params)

    b_matrix = create_matrix(nuclides, name="b")

    As = []
    bs = []
    symbols = []
    for N, Z in zip(Ns_valley, Zs_valley):
        if N < b_matrix.shape[0] and Z < b_matrix.shape[1] and b_matrix[N, Z] != 0:
            As.append(N + Z)
            bs.append(b_matrix[N, Z])
            symbols.append(nuclides[[nuclide['N'] == N and nuclide['Z'] == Z for nuclide in nuclides].index(True)]['symbol'])

    plt.plot(As, bs, 'o-', color='black')

    oldSymbol = ""
    oldA = -1
    oldB = -1
    for A, b, symbol in zip(As, bs, symbols):          
        b += 0.05
        if symbol != oldSymbol:
            if A - oldA < 3 and abs(b - oldB) < 0.1:
                b = 0.08 + oldB

            plt.text(A, b, symbol, fontsize=10, ha='center', va='bottom', color='blue')
            oldA = A
            oldB = b

        oldSymbol = symbol  

    plt.title("Nuclei along the Valley of Stability")
    plt.xlabel('A')
    plt.ylabel('B/A [MeV]')
    plt.grid(alpha=0.5)
    plt.show()


nuclides = import_masses()
most_bound_nuclei(nuclides)

import_spins(nuclides)

best_params = compare_fits(nuclides)
plot_drip_lines(*best_params)

b_matrix = create_matrix(nuclides)
plot_heatmap(b_matrix, title="Binding Energy per Nucleon", vmin=6, vmax=9)

spin_matrix = 0.5 * create_matrix(nuclides, name="spin")
plot_heatmap(spin_matrix, title="Ground-state spins", label="J", vmin=-0.25, vmax=6.25, elements=13)
plot_iso_lines(nuclides, spin_matrix, title="Nuclear spin", Z=20, N=20)
plot_iso_lines(nuclides, spin_matrix, title="Nuclear spin", Z=82, N=126)

parity_matrix = create_matrix(nuclides, name="parity")
plot_heatmap(parity_matrix, title="Nuclear Parities", show_colorbar=False, vmin=-2.5, vmax=2.5, elements=5)
plot_iso_lines(nuclides, parity_matrix, title="Nuclear parity", Z=20, N=20)
plot_iso_lines(nuclides, parity_matrix, title="Nuclear parity", Z=82, N=126)

plot_isotope_lines(nuclides, b_matrix, title="Binding energy per nucleon", Zs=[26])
plot_beta_lines(b_matrix, title="Beta lines", As=[133,134])

alpha = alpha_decay(nuclides)
plot_heatmap(alpha, title="Possible Alpha Decay", show_colorbar=False)

beta = beta_decay(nuclides)
plot_heatmap(beta, title="Possible Beta Decay", show_colorbar=False)

deviation = Bethe_Weizsacker_deviation(nuclides, *best_params)
plot_heatmap(deviation, title="Deviation Experiment - Bethe-Weizsäcker", label='Bexp/A - Btheor/A [MeV]', vmin=-10, vmax=10)

SE = separation(nuclides, 0, 1)
plot_heatmap(SE, title="1p separation Energy", label="S1p [MeV]", vmin=0, vmax=15)

SE = separation(nuclides, 1, 0)
plot_heatmap(SE, title="1n separation Energy", label="S1n [MeV]", vmin=0, vmax=15)

SE = separation(nuclides, 0, 2)
plot_heatmap(SE, title="2p separation Energy", label="S2p [MeV]", vmin=0, vmax=25)

SE = separation(nuclides, 2, 0)
plot_heatmap(SE, title="2n separation Energy", label="S2n [MeV]", vmin=0, vmax=25)
plot_isotope_lines(nuclides, SE, title="2n separation energy", label="S2n [MeV]")

SE = separation(nuclides, 2, 2)
plot_heatmap(SE, title="α separation Energy", label="Sα [MeV]", vmin=-10, vmax=20)

plot_nuclei_along_valley_of_stability(nuclides, best_params)