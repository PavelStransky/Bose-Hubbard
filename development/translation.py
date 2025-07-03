import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from qutip import *

def generate_fixed_N_states(N_sites, N_max, N_total):
    """All Fock states with fixed total particle number."""
    return [state for state in product(range(N_max + 1), repeat=N_sites) if sum(state) == N_total]

def translate_state(state, shift):
    """Cyclic translation (periodic BC)."""
    L = len(state)
    return tuple(state[(i - shift) % L] for i in range(L))

def generate_translation_orbits(states):
    """Group Fock states into translation orbits."""
    seen = set()
    orbits = []
    for s in states:
        if s in seen:
            continue
        orbit = set(translate_state(s, r) for r in range(len(s)))
        seen.update(orbit)
        orbits.append(sorted(orbit))  # canonical ordering
    return orbits

def construct_momentum_basis(orbits, k_sector, N_sites):
    """Project each orbit to momentum eigenstate with momentum k."""
    momentum_basis = []
    for orbit in orbits:
        L = N_sites
        phase_sum = np.zeros((len(orbit[0]),), dtype=complex)  # dummy vector for size

        vec = None
        norm = 0.0
        for r, s in enumerate(orbit):
            phase = np.exp(-2j * np.pi * k_sector * r / L)
            ket = tensor([basis(N_max + 1, n) for n in s])
            if vec is None:
                vec = phase * ket
            else:
                vec = vec + phase * ket
            norm += np.abs(phase) ** 2
        vec = vec.unit()
        momentum_basis.append(vec)
    return momentum_basis

def bose_hubbard_block(N_sites, N_max, N_total, J, U, mu, k_sector):
    """Construct Hamiltonian block for given momentum sector."""
    all_states = generate_fixed_N_states(N_sites, N_max, N_total)
    orbits = generate_translation_orbits(all_states)
    momentum_basis = construct_momentum_basis(orbits, k_sector, N_sites)

    dim = len(momentum_basis)
    H = Qobj(np.zeros((dim, dim), dtype=complex))

    # Create a_ops
    a_ops = []
    for i in range(N_sites):
        op_list = [qeye(N_max + 1)] * N_sites
        op_list[i] = destroy(N_max + 1)
        a_ops.append(tensor(op_list))

    # Build Hamiltonian matrix in momentum basis
    for i in range(dim):
        for j in range(dim):
            bra = momentum_basis[i].dag()
            ket = momentum_basis[j]
            element = 0.0

            # Hopping with PBC
            for k in range(N_sites):
                a_dag = a_ops[k]
                a = a_ops[(k + 1) % N_sites]
                element += -J * (bra * a_dag.dag() * a * ket).data[0, 0]
                element += -J * (bra * a.dag() * a_dag * ket).data[0, 0]

            # On-site and chemical potential
            if i == j:
                occs = orbits[i][0]  # representative state
                onsite_energy = sum((U / 2) * n * (n - 1) - mu * n for n in occs)
                element += onsite_energy

            H[i, j] = element
    return H

# Parameters
N_sites = 3
N_max = 20
J = 1.0
U = -5.0
mu = 0.0
N_total = 20

# Build and plot spectrum by momentum sector
k_vals = list(range(N_sites))  # momentum sectors (mod N_sites)
spectrum_by_k = {}

for k in k_vals:
    H_k = bose_hubbard_block(N_sites, N_max, N_total, J, U, mu, k)
    eigvals = np.sort(H_k.eigenenergies())
    spectrum_by_k[k] = eigvals

# Plot
plt.figure(figsize=(8, 5))
for k, eigs in spectrum_by_k.items():
    plt.plot([k] * len(eigs), eigs, 'bo')

plt.xticks(k_vals)
plt.xlabel("Momentum Sector $k$")
plt.ylabel("Energy")
plt.title(f"Bose-Hubbard Spectrum (N={N_total}, L={N_sites})")
plt.grid(True)
plt.tight_layout()
plt.show()