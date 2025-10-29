import numpy as np
import matplotlib.pyplot as plt
# from spheroid import calculate_and_plot, plot_magic_numbers, fname
from nilsson_cartesian import calculate_and_plot, plot_magic_numbers, fname
# from nilsson_spherical import calculate_and_plot, plot_magic_numbers, fname
from alive_progress import alive_bar

def total_energy(energies, particles):
    result = [sum(energies[i, 0:particles]) for i in range(energies.shape[0])]
    return np.array(result)

def analyze_shapes(deltas, energies, max_particles=100):
    prolate = [0]
    oblate = [0]
    spherical = [0]
    particles = [0]

    min_deltas = [0]
    min_energy = [0]

    for p in range(1, max_particles + 1):
        e = total_energy(energies, p)
        particles.append(p)
        prolate.append(prolate[-1])
        oblate.append(oblate[-1])
        spherical.append(spherical[-1])

        if deltas[np.argmin(e)] < 0:
            oblate[-1] += 1
        elif deltas[np.argmin(e)] > 0:
            prolate[-1] += 1
        else:     
            spherical[-1] += 1

        min_deltas.append(deltas[np.argmin(e)])
        min_energy.append(np.min(e))

    energy_differences = np.diff(min_energy) 

    plt.plot(particles[1:], energy_differences)
    plot_magic_numbers()
    plt.xlabel("Number of particles")
    plt.ylabel("S1")
    plt.title("One-particle separation energy")
    plt.show()

    plt.plot(particles, prolate, label="Prolate", color='blue')
    plt.plot(particles, oblate, label="Oblate", color='red')
    plt.plot(particles, spherical, label="Spherical", color='green')
    plot_magic_numbers()
    plt.xlabel("Number of particles")
    plt.ylabel("Number of occurrences")
    plt.title("Predicted Equilibrium Shapes")
    plt.legend()
    plt.show()

    plt.plot(particles, min_deltas)
    plot_magic_numbers()
    plt.xlabel("Number of particles")
    plt.ylabel("Deformation δ")
    plt.title("Equilibrium Deformation δ")
    plt.show()

""" Main analysis"""
if __name__ == "__main__":
    try:
        delta_energies = np.loadtxt(fname)
        print(f"Loaded precomputed energies from '{fname}'")
    except FileNotFoundError:
        delta_energies = calculate_and_plot()
        np.savetxt(fname, delta_energies)

    deltas = delta_energies[:,0]
    energies = delta_energies[:,1:]

    particles = 15
    plt.plot(deltas, total_energy(energies, particles), color='black', linewidth=2)
    plt.xlabel("Deformation δ")
    plt.ylabel(f"E")
    plt.title(f"Total energy for {particles} particles")
    plt.show()

    analyze_shapes(deltas, energies, max_particles=200)