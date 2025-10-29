import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar

def calculate_and_plot(num_levels=1000, num_delta=1001):
    """ Calculate and plot the energy levels of a 3D anisotropic harmonic oscillator"""
    deltas = np.linspace(-0.5, 0.5, num_delta)
    delta_energies = np.zeros((num_delta, num_levels + 1))
    delta_energies[:,0] = deltas

    max_E = np.sqrt(num_levels)
    max_n = int(max_E / 2)

    with alive_bar(len(deltas), title="Computing Energies") as bar:
        for j, delta in enumerate(deltas):
            energies = []

            for n_x in range(max_n + 1):
                for n_y in range(max_n + 1):
                    for n_z in range(max_n + 1):
                        n = n_x + n_y + n_z
                        energy = (n_x + 0.5) * np.exp(delta) + (n_y + 0.5) * np.exp(delta) + (n_z + 0.5) * np.exp(-2 * delta)
                        if energy <= max_E:
                            energies.append(energy)
        
            energies = np.array(sorted(energies)[:num_levels])
            delta_energies[j, 1:] = energies
            bar()

    plt.plot(deltas, delta_energies[:, 1:], color='blue', linewidth=1)
    plt.xlabel("Deformation δ")
    plt.ylabel("Energy (ℏω units)")
    plt.ylim(0, 10)
    plt.title("3D Anisotropic Harmonic Oscillator in Cartesian Basis")
    plt.show()

    return delta_energies

def plot_magic_numbers():
    """ Plot vertical lines at magic numbers 
        (Magic numbers were determined from the one-particle separation energy plot) """
    magic_numbers = [10, 20, 35, 56, 84, 120, 165]
    for mn in magic_numbers:
        plt.axvline(mn, color='gray', linestyle='--', linewidth=0.8)
        plt.text(mn + 1, plt.ylim()[1] * 0.9, str(mn), rotation=90, color='gray')

fname = "nilsson_spectrum_cartesian.dat"
