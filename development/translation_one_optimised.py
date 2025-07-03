import numpy as np
from itertools import product
from scipy.sparse import csc_matrix
from alive_progress import alive_bar, alive_it
from qutip import *
import time

# -----------------------------
# Parameters
# -----------------------------
N_sites = 4        # number of lattice sites
N_max = 20        # max bosons per site
N_total = 20      # fixed total boson number
k_sector = 1       # momentum sector (integer from 0 to N_sites-1)

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

print(f"Constructing momentum eigenstates for k = {k_sector}...")
basis_k = [momentum_eigenstate(orbit, k_sector, N_sites) for orbit in alive_it(orbits)]

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
# Construct Hamiltonian matrix in momentum basis (projected method)
# -----------------------------

# Step 1: Build projection matrix B from full Fock space to momentum basis
start_time = time.time()
dim_full = a_ops[0].shape[0]
dim_reduced = len(basis_k)
print(f"Building projection matrix B from full Fock space ({dim_full}) to reduced basis ({dim_reduced})...")
B = np.column_stack([psi.full().flatten() for psi in alive_it(basis_k)])  # shape: (dim_full, dim_reduced)

# Step 2: Initialize sparse Hamiltonian in reduced basis
H_mat = np.zeros((dim_reduced, dim_reduced), dtype=complex)

# Step 3: Add hopping terms
print("Projecting hopping terms...")
with alive_bar(N_sites) as bar:
    for l in range(N_sites):
        aa = a_ops[l].dag() * a_ops[(l + 1) % N_sites]
        aa_dag = aa.dag()
        aa_total = (-J * aa - J * aa_dag).data.as_scipy()  # full Fock space operator, in sparse format

        B_sparse = csc_matrix(B)

        # Project to momentum basis: H_k = B† O B        
        H_mat += B_sparse.getH() @ aa_total @ B_sparse
        bar()

# Step 4: Add on-site interaction and chemical potential (diagonal)
print("Adding on-site and chemical potential terms...")
with alive_bar(dim_reduced) as bar:
    for i in range(dim_reduced):
        state = orbits[i][0]  # representative Fock state of the orbit
        energy = 0.0
        for n in state:
            energy += (U / N_total) * n * (n - 1) - mu * n
        H_mat[i, i] += energy
        bar()

# Step 5: Wrap into a QuTiP Qobj
H = Qobj(H_mat)

# -----------------------------
# Diagonalize and print spectrum
# -----------------------------
eigenvals = np.sort(H.eigenenergies()) / N_total
print(f"\nEigenvalues for N_total = {N_total}, k = {k_sector}:\n", eigenvals)

np.savetxt("output.csv", eigenvals)