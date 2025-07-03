import numpy as np
from itertools import product
from alive_progress import alive_bar
from qutip import *
import time

# -----------------------------
# Parameters
# -----------------------------
N_sites = 4       # number of lattice sites
N_max = 20          # max bosons per site
N_total = 20        # fixed total boson number
k_sector = 3       # momentum sector (integer from 0 to N_sites-1)

J = 1.0            # hopping amplitude
U = -5.0            # on-site interaction
mu = 0.0           # chemical potential

# -----------------------------
# Generate all Fock states with total particle number N_total
# -----------------------------
def generate_states(N_sites, N_max, N_total):
    return [tuple(s) for s in product(range(N_max+1), repeat=N_sites) if sum(s) == N_total]

def translate_state(state, shift):
    L = len(state)
    return tuple(state[(i - shift) % L] for i in range(L))

# -----------------------------
# Find orbit representatives and construct momentum eigenstates
# -----------------------------
def get_orbits(states):
    seen = set()
    orbits = []
    for s in states:
        if s in seen:
            continue
        orbit = {translate_state(s, r) for r in range(len(s))}
        seen.update(orbit)
        orbits.append(sorted(orbit))
    return orbits

def momentum_eigenstate(orbit, k, N_sites):
    """Construct normalized momentum eigenstate from translation orbit."""
    psi = None
    for r, state in enumerate(orbit):
        phase = np.exp(-2j * np.pi * k * r / N_sites)
        ket = tensor([basis(N_max+1, n) for n in state])
        psi = phase * ket if psi is None else psi + phase * ket
    return psi.unit()

# -----------------------------
# Build momentum basis
# -----------------------------
start_time = time.time()
all_states = generate_states(N_sites, N_max, N_total)
print(f"Generated {len(all_states)} states in {time.time() - start_time:.2f} seconds.")

start_time = time.time()
orbits = get_orbits(all_states)
print(f"Found {len(orbits)} orbits in {time.time() - start_time:.2f} seconds.")

start_time = time.time()
basis_k = [momentum_eigenstate(orbit, k_sector, N_sites) for orbit in orbits]
print(f"Constructed {len(basis_k)} momentum eigenstates in {time.time() - start_time:.2f} seconds.")

# -----------------------------
# Create site operators
# -----------------------------
a_ops = []
n_ops = []
with alive_bar(N_sites) as bar:
    for i in range(N_sites):
        a_list = [qeye(N_max+1)] * N_sites
        n_list = [qeye(N_max+1)] * N_sites
        a_list[i] = destroy(N_max+1)
        n_list[i] = num(N_max+1)
        a_ops.append(tensor(a_list))
        n_ops.append(tensor(n_list))
        bar()

# -----------------------------
# Construct Hamiltonian matrix in momentum basis
# -----------------------------
dim = len(basis_k)
H_mat = np.zeros((dim, dim), dtype=complex)

for l in range(N_sites):
    aa = a_ops[l].dag() * a_ops[(l+1) % N_sites]
    aa_dag = aa.dag()

    with alive_bar(dim) as bar:
        for i in range(dim):
            bra = basis_k[i].dag()

            braaa = bra * aa
            braaa_dag = bra * aa_dag

            for j in range(i, dim):
                ket = basis_k[j]
                h_elem = 0.0

                # Hopping with periodic boundary conditions
                h_elem += -J * (braaa * ket)
                h_elem += -J * (braaa_dag * ket)

                # On-site terms (only diagonal)
                if i == j and l == 0:
                    state = orbits[i][0]
                    for n in state:
                        h_elem += (U/N_total) * n * (n - 1) - mu * n

                H_mat[i, j] += h_elem

                if i != j:
                    H_mat[j, i] += h_elem.conjugate()  # Ensure Hermitian matrix
            bar()

H = Qobj(H_mat)

# -----------------------------
# Diagonalize and print spectrum
# -----------------------------
eigenvals = np.sort(H.eigenenergies()) / N_total
print(f"\nEigenvalues for N_total = {N_total}, k = {k_sector}:\n", eigenvals)
