import numpy as np
import matplotlib.pyplot as plt
from qutip import *

def bose_hubbard_hamiltonian(N_sites, N_max, J, U, mu):
    # Create annihilation operators for each site
    a_list = []
    for i in range(N_sites):
        op_list = [qeye(N_max+1)] * N_sites
        op_list[i] = destroy(N_max+1)
        a_list.append(tensor(op_list))

    # Hopping term: -J * sum(a_i^\dagger a_{i+1} + h.c.)
    H_hop = 0
    for i in range(N_sites - 1):  # open boundary conditions
        H_hop += -J * (a_list[i].dag() * a_list[i+1] + a_list[i+1].dag() * a_list[i])

    H_hop += -J * (a_list[N_sites - 1].dag() * a_list[0] + a_list[0].dag() * a_list[N_sites - 1])  # periodic boundary conditions

    # Interaction term: (U/2) * sum n_i (n_i - 1)
    H_int = 0
    for a in a_list:
        n = a.dag() * a
        H_int += (U/N_sites) * n * (n - 1)

    # Chemical potential term: -mu * sum n_i
    H_mu = 0
    for a in a_list:
        n = a.dag() * a
        H_mu += -mu * n

    H = H_hop + H_int + H_mu
    return H

# Parameters
N_sites = 3         # number of sites
N_max = 30          # max bosons per site
J = 1.0             # hopping amplitude
U = -5.0             # interaction strength
mu = 0.0            # chemical potential

# Hamiltonian and diagonalization
H = bose_hubbard_hamiltonian(N_sites, N_max, J, U, mu)
eigenvalues = H.eigenenergies()

# Plotting
plt.figure(figsize=(6,4))
plt.plot(np.arange(len(eigenvalues)), eigenvalues, 'bo')
plt.xlabel("State index")
plt.ylabel("Energy")
plt.title("Bose-Hubbard Spectrum")
plt.grid(True)
plt.tight_layout()
plt.show()