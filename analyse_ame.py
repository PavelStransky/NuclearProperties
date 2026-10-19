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
            b = line[54:67]

            # We skip nonexperimental values marked by the # sign
            if b.find('#') >= 0:
                continue

            # We convert keVs into MeVs
            nuclide["b"] = float(b) / 1000
            nuclide["berror"] = float(line[68:78]) / 1000

            # Binding energy (capital B)
            nuclide["B"] = nuclide["b"] * nuclide["A"]
            nuclide["Berror"] = nuclide["berror"] * nuclide["A"]

            # We calculate the nuclear mass (can be also used the last column from the data file)
            nuclide["M"] = Mn * nuclide["N"] + Mp * nuclide["Z"] - nuclide["B"]

            nuclides.append(nuclide)

    return nuclides


def import_spins(nuclides):
    """ Reads ground-state spins (stored as 2J) and parities from the data file and adds them to the nuclides array.
        Unknown values are NaN.
    """
    ignore_characters = ['#', '(', ')', ' ', '*']

    nuclides_by_AZ = {(nuclide["A"], nuclide["Z"]): nuclide for nuclide in nuclides}

    for nuclide in nuclides:
        nuclide["spin"] = np.nan
        nuclide["parity"] = np.nan

    with open(r"nubase_4.mas20.txt") as file:
        for line in file:
            # We skip the header
            if line.startswith('#'):
                continue

            A = int(line[0:3])
            Z = int(line[4:7])
            iso = int(line[7:8])

            # We take only ground states (iso > 0 are isomers, excited levels, resonances and isobaric analog states)
            if iso != 0 or (A, Z) not in nuclides_by_AZ:
                continue

            # Spin and parity without the isospin (T=...)
            spin_str = line[88:102].split("T=")[0]

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

            parity = np.nan
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

            nuclide = nuclides_by_AZ[(A, Z)]
            nuclide["spin"] = spin
            nuclide["parity"] = parity

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
    """ Sorts the nuclides array according to the given criterion """
    return sorted(nuclides, key=lambda nuclide: nuclide[name])


def create_matrix(nuclides, name="b"):
    """ Returns the element with the given name as a 2D array indexed [N, Z] (NaN where there are no data) """
    max_N, max_Z, _ = max_nza(nuclides)

    result = np.full((max_N + 1, max_Z + 1), np.nan)

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


def plot_heatmap(data_matrix, title="", vmin=None, vmax=None, show_colorbar=True, label='B/A [MeV]', elements=2048, categories=None, under=None):
    """ Plots a heatmap of the given data matrix (NaN values are not shown)
        categories : labels of the integer values 1, 2, ..., n (for categorical data)
        under : color of the values below vmin
    """
    # Mask missing values
    data_masked = np.ma.masked_invalid(data_matrix)

    if categories is not None:
        elements = len(categories)
        vmin, vmax = 0.5, len(categories) + 0.5

    # Create colormap that leaves masked values as white
    cmap = colormap(elements)
    cmap.set_bad(color='white')  # masked values appear white

    if under is not None:
        cmap.set_under(color=under)

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
        colorbar = plt.colorbar(label=label, extend='min' if under is not None else 'neither')
        if categories is not None:
            colorbar.set_ticks(range(1, len(categories) + 1), labels=categories)

    plt.title(title)

    plt.xlim(0, data_matrix.shape[0])
    plt.ylim(0, data_matrix.shape[1])

    plt.xlabel('N')
    plt.ylabel('Z')
    plt.show()


def Bethe_Weizsacker(N, Z, a_v=15.76, a_s=17.81, a_c=0.711, a_a=23.7, delta=34.0, delta_exp=3/4):
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
    for i, nuclide in enumerate(nuclides_sorted[:num], start=1):
        print(f"{i}: {nuclide['symbol']} Z={nuclide['Z']}, N={nuclide['N']}, BA={nuclide['b']:.4f} MeV")


def separation(nuclides, N_diff=0, Z_diff=0):
    """ Calculate separation energy matrix
        N_diff : difference in neutron number (e.g., 2 for 2n separation energy)
        Z_diff : difference in proton number (e.g., 2 for 2p separation energy)
    """
    M_matrix = create_matrix(nuclides, name="M")
    result = np.full_like(M_matrix, np.nan)

    M_diff = Z_diff * Mp + N_diff * Mn

    if Z_diff != 0 and N_diff != 0 and not np.isnan(M_matrix[N_diff, Z_diff]):
        M_diff = M_matrix[N_diff, Z_diff]  # If we separate an existing nucleus, we use its mass

    for N in range(N_diff, M_matrix.shape[0]):
        for Z in range(Z_diff, M_matrix.shape[1]):
            # A missing mass (NaN) gives NaN separation energy
            result[N, Z] = M_matrix[N - N_diff, Z - Z_diff] - M_matrix[N, Z] + M_diff

    return result


def plot_beta_lines(matrix, As=range(30, 40), title="", label="B/A [MeV]"):
    """ Plot Beta lines (A = const) for given A range """
    Zs = np.array(range(matrix.shape[1]), dtype=int)

    for A in As:
        Nx = -Zs + A
        mask = (Nx >= 0) & (Nx < matrix.shape[0])

        Nx = Nx[mask]
        if len(Nx) == 0:
            continue

        Zx = Zs[mask]

        line = matrix[Nx, Zx]

        if np.all(np.isnan(line)):
            continue

        mask = ~np.isnan(line)

        Zx = Zx[mask]
        line = line[mask]

        plt.plot(Zx, line, 'o-', label=f'A={A}')

    plt.title(title)
    plt.xlabel('Z')
    plt.ylabel(label)
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.show()


def plot_isotope_lines(nuclides, matrix, Zs=None, title="", label="B/A [MeV]"):
    """ Plot isotope lines for given Z range """

    if Zs is None:
        Zs = range(matrix.shape[1])

    for Z in Zs:
        line = matrix[:, Z]

        if np.all(np.isnan(line)):
            continue

        Ns = np.array(range(matrix.shape[0]))

        mask = ~np.isnan(line)

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

    mask = ~np.isnan(line)

    Ns = Ns[mask]
    line = line[mask]

    plt.plot(Ns, line, 'o-', alpha=0.7, label=f'isotopes Z={Z}')

    line = matrix[N, :]
    Zs = np.array(range(matrix.shape[1]))

    mask = ~np.isnan(line)

    Zs = Zs[mask]
    line = line[mask]

    plt.plot(Zs, line, 'o-', alpha=0.7, label=f'isotones N={N}')

    plt.title(title)
    plt.xlabel('Z,N')
    plt.ylabel(label)
    plt.legend()
    plt.show()


def compare_fits(nuclides, cut=20):
    """ Compare different fits to the Bethe-Weizsäcker formula (first for all nuclides, then only for A >= cut).
        Returns the parameters of the best fit for A >= cut including the pairing exponent,
        so that they can be passed directly as Bethe_Weizsacker(N, Z, *params)
    """
    # Free nucleons (A = 1) are not bound systems
    nuclides = [nuclide for nuclide in nuclides if nuclide['A'] > 1]

    Ns = np.array([nuclide['N'] for nuclide in nuclides])
    Zs = np.array([nuclide['Z'] for nuclide in nuclides])
    Bs = np.array([nuclide['B'] for nuclide in nuclides])

    def BW_fit_B(NZ, a_v=15.76, a_s=17.81, a_c=0.711, a_a=23.7, delta=33.5):
        """ Wrapper for Bethe-Weizsäcker to fit B """
        N, Z = NZ
        return Bethe_Weizsacker(N, Z, a_v, a_s, a_c, a_a, delta)

    def BW_fit_b(NZ, a_v=15.76, a_s=17.81, a_c=0.711, a_a=23.7, delta=33.5):
        """ Wrapper for Bethe-Weizsäcker to fit the binding energy per nucleon """
        N, Z = NZ
        return Bethe_Weizsacker(N, Z, a_v, a_s, a_c, a_a, delta) / (N + Z)

    def BW_fit_B_fixedAC(NZ, a_v=15.76, a_s=17.81, a_a=23.7, delta=33.5):
        """ Wrapper for Bethe-Weizsäcker to fit B with the Coulomb coefficient fixed to a_c = 0.711 MeV """
        N, Z = NZ
        return Bethe_Weizsacker(N, Z, a_v, a_s, 0.711, a_a, delta)

    def BW_fit_B_pairing12(NZ, a_v=15.76, a_s=17.81, a_c=0.711, a_a=23.7, delta=33.5):
        """ Wrapper for Bethe-Weizsäcker to fit B with pairing term proportional to A^(-1/2) """
        N, Z = NZ
        return Bethe_Weizsacker(N, Z, a_v, a_s, a_c, a_a, delta, delta_exp=1/2)

    def rms(residuals):
        """ Root-mean-square deviation """
        return np.sqrt(np.mean(residuals**2))

    def fit(N, Z, B, params, fit_function, per_nucleon=False):
        """ Fits B (or B/A if per_nucleon) and prints the RMS deviation of B """
        NZ = np.vstack((N, Z))
        A = N + Z

        p_values, covariance = curve_fit(fit_function, NZ, B / A if per_nucleon else B, p0=list(params.values()))
        res = B - fit_function(NZ, *p_values) * (A if per_nucleon else 1)

        print(f"{fit_function.__name__} (RMS={rms(res):.3f} MeV)", [f"{name}={value:.3f}+{error:.3f}" for name, value, error in zip(params.keys(), p_values, np.sqrt(np.diag(covariance)))])

        return p_values

    for A_min in [2, cut]:
        mask = Ns + Zs >= A_min
        N, Z, B = Ns[mask], Zs[mask], Bs[mask]

        print(f"\nFitting {len(B)} nuclei with A >= {A_min}")

        print(f"Wiki (LS2) (RMS={rms(B - Bethe_Weizsacker(N, Z)):.3f} MeV)")

        params = {"a_v": 15.75, "a_s": 17.8, "a_c": 0.711, "a_a": 23.7, "delta": 34}
        fit(N, Z, B, params, BW_fit_b, per_nucleon=True)

        params = {"a_v": 15.75, "a_s": 17.8, "a_c": 0.711, "a_a": 23.7, "delta": 34}
        fit(N, Z, B, params, BW_fit_B)

        params = {"a_v": 15.75, "a_s": 17.8, "a_a": 23.7, "delta": 34}
        fit(N, Z, B, params, BW_fit_B_fixedAC)

        params = {"a_v": 15.75, "a_s": 17.8, "a_c": 0.711, "a_a": 23.7, "delta": 34}
        best_fit = fit(N, Z, B, params, BW_fit_B_pairing12)

    # The best fit has the pairing term proportional to A^(-1/2)
    return [*best_fit, 1/2]

def Bethe_Weizsacker_deviation(nuclides, *params):
    """ Calculate deviation matrix from Bethe-Weizsäcker formula. """
    max_N, max_Z, _ = max_nza(nuclides)

    difference_matrix = np.full((max_N + 1, max_Z + 1), np.nan)

    for nuclide in nuclides:
        N = nuclide['N']
        Z = nuclide['Z']
        B = nuclide['B']
        difference_matrix[N, Z] = B - Bethe_Weizsacker(N, Z, *params)

    return difference_matrix


def neutron_drip_line(*params):
    """ Neutron drip line """
    Ns = range(1, 300)

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


def beta_stability(*params):
    """ Valley of beta stability: for each A, the nucleus with the minimal atomic mass
        M = Z M_H + N M_n - B, i.e., with the maximal B + Z (M_n - M_H)
    """
    Zs = np.array(range(1, 120), dtype=int)

    drip_line = []
    Ns = []

    for A in range(1, 300):
        Nx = -Zs + A
        mask = Nx > 0

        Nx = Nx[mask]
        if len(Nx) == 0:
            continue

        # Hydrogen atom mass M_H = M_p + M_e (electron binding neglected)
        B = Bethe_Weizsacker(Nx, Zs[mask], *params) + Zs[mask] * (Mn - Mp - Me)
        max_i = np.argmax(B)
        if max_i < len(B) - 1:
            Ns.append(Nx[max_i])
            drip_line.append(Zs[mask][max_i])

    return np.array(Ns, dtype=int), np.array(drip_line, dtype=int)


def plot_drip_lines(*best_params):
    plt.plot(*neutron_drip_line(*best_params), color='blue', linestyle='-', linewidth=1, label='Neutron Drip Line')
    plt.plot(*proton_drip_line(*best_params), color='green', linestyle='-', linewidth=1, label='Proton Drip Line')
    plt.plot(*beta_stability(*best_params), color='black', linestyle='-', linewidth=1, label='Valley of Stability')
    plt.legend()


def alpha_decay(nuclides):
    """ Finds all nuclides that can decay via alpha process:
        1 = alpha decay energetically possible (Q_alpha > 0), 2 = forbidden,
        NaN = cannot be decided (mass of the daughter nucleus not measured)
    """

    M_matrix = create_matrix(nuclides, name="M")

    # Helium - alpha particle
    M_alpha = M_matrix[2, 2]

    alpha = np.full_like(M_matrix, np.nan)

    max_N, max_Z = M_matrix.shape

    for N in range(2, max_N):
        for Z in range(2, max_Z):
            Q_alpha = M_matrix[N, Z] - M_matrix[N - 2, Z - 2] - M_alpha

            # Parent or daughter not measured
            if np.isnan(Q_alpha):
                continue

            alpha[N, Z] = 1 if Q_alpha > 0 else 2

    return alpha


def beta_decay(nuclides):
    """ Finds all nuclides that can decay via beta process:
        1 = beta-minus, 2 = electron capture (and beta-plus if Q_EC > 2 Me), 3 = beta-stable,
        NaN = cannot be decided (mass of the daughter nucleus not measured)
    """

    M_matrix = create_matrix(nuclides, name="M")

    beta = np.full_like(M_matrix, np.nan)

    max_N, max_Z = M_matrix.shape

    def mass(N, Z):
        """ Mass of the nucleus (NaN if not measured) """
        if N < max_N and Z < max_Z:
            return M_matrix[N, Z]
        return np.nan

    for N in range(max_N):
        for Z in range(max_Z):
            if np.isnan(M_matrix[N, Z]):
                continue

            # Q-values from nuclear masses (electron binding energies neglected);
            # -inf if the process is impossible (no neutron or proton to convert)

            # beta-minus: (N, Z) -> (N - 1, Z + 1) + e- + antineutrino
            Q_betaminus = M_matrix[N, Z] - mass(N - 1, Z + 1) - Me if N > 0 else -np.inf

            # electron capture: (N, Z) + e- -> (N + 1, Z - 1) + neutrino
            Q_EC = M_matrix[N, Z] - mass(N + 1, Z - 1) + Me if Z > 0 else -np.inf

            if Q_betaminus > 0:
                beta[N, Z] = 1
            elif Q_EC > 0:
                beta[N, Z] = 2
            elif not np.isnan(Q_betaminus) and not np.isnan(Q_EC):
                beta[N, Z] = 3

    return beta


def plot_nuclei_along_beta_stability(nuclides, params):
    Ns_valley, Zs_valley = beta_stability(*params)

    b_matrix = create_matrix(nuclides, name="b")

    As = []
    bs = []
    symbols = []
    for N, Z in zip(Ns_valley, Zs_valley):
        if N < b_matrix.shape[0] and Z < b_matrix.shape[1] and not np.isnan(b_matrix[N, Z]):
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

""" Main analysis """
if __name__ == "__main__":
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

    plot_isotope_lines(nuclides, b_matrix, title="Binding energy per nucleon", Zs=[26])
    plot_beta_lines(b_matrix, title="Beta lines", As=[133,134])

    alpha = alpha_decay(nuclides)
    plot_heatmap(alpha, title="Possible Alpha Decay", label="", categories=[r"$Q_\alpha > 0$", r"$Q_\alpha < 0$"])

    beta = beta_decay(nuclides)
    plot_drip_lines(*best_params)
    plot_heatmap(beta, title="Possible Beta Decay", label="", categories=[r"$\beta^-$", r"EC / $\beta^+$", "stable"])

    deviation = Bethe_Weizsacker_deviation(nuclides, *best_params)
    plot_heatmap(deviation, title="Deviation Experiment - Bethe-Weizsäcker", label='Bexp - Btheor [MeV]', vmin=-10, vmax=10)

    # Unbound nuclei (negative separation energy) in gray
    SE = separation(nuclides, 0, 1)
    plot_heatmap(SE, title="1p separation Energy", label="S1p [MeV]", vmin=0, vmax=15, under='gray')

    SE = separation(nuclides, 1, 0)
    plot_heatmap(SE, title="1n separation Energy", label="S1n [MeV]", vmin=0, vmax=15, under='gray')

    SE = separation(nuclides, 0, 2)
    plot_heatmap(SE, title="2p separation Energy", label="S2p [MeV]", vmin=0, vmax=25, under='gray')

    SE = separation(nuclides, 2, 0)
    plot_heatmap(SE, title="2n separation Energy", label="S2n [MeV]", vmin=0, vmax=25, under='gray')
    plot_isotope_lines(nuclides, SE, title="2n separation energy", label="S2n [MeV]")

    SE = separation(nuclides, 2, 2)
    plot_heatmap(SE, title="α separation Energy", label="Sα [MeV]", vmin=-10, vmax=20)

    plot_nuclei_along_beta_stability(nuclides, best_params)