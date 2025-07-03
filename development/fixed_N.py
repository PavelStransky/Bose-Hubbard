import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from qutip import *

def number_states_in_sector(N_sites, N_max, N_total):
    """
    Generate all Fock states with total occupation N_total.
    Each site can hold up to N_max bosons.
    """
    all_states = []
    for occ in product(range(N_max + 1), repeat=N_sites):
        if sum(occ) == N_total:
            all_states.append(occ)
    return all_states

def create_basis_states(state_list, N_max):
    """
    Convert a list of occupation number tuples into QuTiP ket vectors.
    """
    basis_states = []
    for state in state_list:
        kets = [basis(N_max + 1, n) for n in state]
        basis_states.append(tensor(kets))
    return basis_states

def bose_hubbard_sector_hamiltonian(N_sites, N_max, J, U, mu, N_total):
    """
    Build the Hamiltonian restricted to a subspace with total boson number = N_total
    """
    # Generate Fock basis with fixed total number of bosons
    state_list = number_states_in_sector(N_sites, N_max, N_total)
    basis_states = create_basis_states(state_list, N_max)

    dim = len(basis_states)
    H = Qobj(np.zeros((dim, dim), dtype=np.complex128))

    # Precompute annihilation and number operators
    a_ops = []
    n_ops = []
    for i in range(N_sites):
        op_list_a = [qeye(N_max+1)] * N_sites
        op_list_n = [qeye(N_max+1)] * N_sites
        op_list_a[i] = destroy(N_max+1)
        op_list_n[i] = num(N_max+1)
        a_ops.append(tensor(op_list_a))
        n_ops.append(tensor(op_list_n))

    # Build Hamiltonian in the full space
    for i in range(dim):
        for j in range(dim):
            bra = basis_states[i].dag()
            ket = basis_states[j]
            element = 0.0

            # Hopping term
            for k in range(N_sites - 1):  # open boundary conditions
                element += -J * (bra * a_ops[k].dag() * a_ops[k+1] * ket)
                element += -J * (bra * a_ops[k+1].dag() * a_ops[k] * ket)

            # Interaction and chemical potential terms (diagonal)
            if i == j:
                for k in range(N_sites):
                    n = state_list[i][k]
                    element += (U / 2) * n * (n - 1) - mu * n

            H[i, j] = element

    return H, state_list

# Parameters
N_sites = 3
N_max = 20
J = 1.0
U = -5.0
mu = 0.0

# Explore several particle number sectors
spectra = {}
for N_total in range(0, N_sites * N_max + 1):
    H_sec, states = bose_hubbard_sector_hamiltonian(N_sites, N_max, J, U, mu, N_total)
    eigvals = np.sort(H_sec.eigenenergies())
    spectra[N_total] = eigvals

# Plot
plt.figure(figsize=(6, 4))
for N_total, eigvals in spectra.items():
    y = eigvals
    x = [N_total] * len(y)
    plt.plot(x, y, 'bo', ms=3)

plt.xlabel("Total Particle Number Sector")
plt.ylabel("Energy")
plt.title("Bose-Hubbard Spectrum by Symmetry Sector")
plt.grid(True)
plt.tight_layout()
plt.show()