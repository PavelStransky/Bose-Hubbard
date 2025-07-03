import numpy as np
from itertools import product
from scipy.sparse import csc_matrix
from alive_progress import alive_it
from qutip import *
import time

# -----------------------------
# Parameters
# -----------------------------
N_sites = 4        # number of lattice sites
N_max = 20        # max bosons per site
N_total = 20      # fixed total boson number
# k = 0       # momentum sector (integer from 0 to N_sites-1)
# parity_sector = -1

J = 1.0            # hopping amplitude
U = -5.0            # on-site interaction
mu = 0.0           # chemical potential

def generate_states(N_sites, N_max, N_total):
    """ Generate all Fock states with total particle number N_total."""

    start_time = time.time()
    result = [tuple(s) for s in product(range(N_max + 1), repeat=N_sites) if sum(s) == N_total]
    print(f"Generated {len(result)} states in {time.time() - start_time:.2f} seconds.")
    return result

def translate_state(state, shift):
    """ Return the translated Fock state tuple by shift. """
    L = len(state)
    return tuple(state[(i - shift) % L] for i in range(L))

def parity_reflect_state(state):
    """Return the parity-reflected Fock state tuple."""
    return tuple(reversed(state))
   
def get_orbits(states):
    """ Find translation orbits and their parity invariants. """

    start_time = time.time()
    
    seen = set()
    orbits = []
    for s in states:
        if s in seen:
            continue
        orbit = {translate_state(s, r) for r in range(len(s))}
        parity_orbit = {parity_reflect_state(state) for state in orbit}

        seen.update(orbit)
        seen.update(parity_orbit)

        parity_invariant = orbit == parity_orbit

        orbit = sorted(orbit)
        parity_orbit = [parity_reflect_state(state) for state in orbit]

        if parity_invariant:
            orbit = [orbit]
        else:
            orbit = [orbit, parity_orbit]

        orbits.append(orbit)

    print(f"Found {len(orbits)} orbits in {time.time() - start_time:.2f} seconds.")

    return orbits

def momentum_eigenstates(N_sites, N_max, orbits):   
    infos = []
    for k in range(N_sites):
        if k == 0 or (k == N_sites // 2 and N_sites % 2 == 0):
            infos.append([k, -1, 0])  # even parity
            infos.append([k, 1, 0])   # odd parity
        else:
            infos.append([k, 0, 0])

    num = len(infos)

    basis_states = [[] for _ in range(num)]
    representative_states = [[] for _ in range(num)]

    print(f"Constructing momentum eigenstates...")
    start_time = time.time()

    for orbit in alive_it(orbits):
        """Construct normalized momentum eigenstates from translation orbit."""
        psi = [None for _ in range(N_sites)]

        for r, state in enumerate(orbit[0]):
            ket = tensor([basis(N_max + 1, n) for n in state])

            for k in range(N_sites):
                phase = np.exp(-2j * np.pi * k * r / N_sites)
                psi[k] = phase * ket if psi[k] is None else psi[k] + phase * ket

        psi_parity = [None for _ in range(N_sites)]
        if len(orbit) > 1:
            for r, state in enumerate(orbit[1]):
                ket = tensor([basis(N_max+1, n) for n in state])

                for k in range(N_sites):
                    phase = np.exp(-2j * np.pi * k * r / N_sites)
                    psi_parity[k] = phase * ket if psi_parity[k] is None else psi_parity[k] + phase * ket

        state = orbit[0][0]  # representative Fock state of the orbit

        for i, info in enumerate(infos):
            k, parity, num = info

            if psi_parity[k] is None:
                if parity != -1:
                    basis_states[i].append(psi[k].unit()), representative_states[i].append(state)
                    info[2] += 1
            
            else:
                even_psi = (psi[k] + psi_parity[k]).unit()
                odd_psi = (psi[k] - psi_parity[k]).unit()

                if parity == 1:  # even parity
                    basis_states[i].append(even_psi), representative_states[i].append(state)
                    info[2] += 1
                elif parity == -1:  # odd parity
                    basis_states[i].append(odd_psi), representative_states[i].append(state)
                    info[2] += 1
                elif parity == 0:  # both parities
                    basis_states[i].append(psi[k]), representative_states[i].append(state)
                    basis_states[i].append(psi_parity[k]), representative_states[i].append(state)
                    info[2] += 2

    total = sum(info[2] for info in infos)
    print(f"Constructed {total} basis states in {time.time() - start_time:.2f} seconds.")

    return infos, basis_states, representative_states


def site_operators(N_sites, N_max):
    """ Create site annihilation and number operators for each site. """
    a_ops = []
    for i in alive_it(range(N_sites)):
        a_list = [qeye(N_max + 1)] * N_sites
        a_list[i] = destroy(N_max + 1)
        a_ops.append(tensor(a_list))
    return a_ops

def build_hamiltonian(N_sites, N_max, N_total, basis_states, representative_states, J, U, mu):
    """ Build the Bose-Hubbard Hamiltonian in the reduced momentum basis_states. """

    print("Building Hamiltonian in reduced momentum basis_states...")
    start_time = time.time()

    a_ops = site_operators(N_sites, N_max)

    dim_full = a_ops[0].shape[0]
    dim_reduced = len(basis_states)

    print(f"Building projection matrix B from full Fock space ({dim_full}) to reduced basis_states ({dim_reduced})...")
    B = np.column_stack([psi.full().flatten() for psi in alive_it(basis_states)])  # shape: (dim_full, dim_reduced)

    # Initialize sparse Hamiltonian in reduced basis_states
    H_mat = np.zeros((dim_reduced, dim_reduced), dtype=complex)

    # Add hopping terms
    print("Projecting hopping terms...")
    for l in alive_it(range(N_sites)):
        aa = a_ops[l].dag() * a_ops[(l + 1) % N_sites]
        aa_dag = aa.dag()
        aa_total = (-J * aa - J * aa_dag).data.as_scipy()  # full Fock space operator, in sparse format

        B_sparse = csc_matrix(B)

        # Project to momentum basis_states
        H_mat += B_sparse.getH() @ aa_total @ B_sparse

    # Add on-site interaction and chemical potential (diagonal)
    print("Adding on-site and chemical potential terms...")
    for i, state in enumerate(alive_it(representative_states)):
        energy = 0.0
        for n in state:
            energy += (U / N_total) * n * (n - 1) - mu * n
        H_mat[i, i] += energy

    print(f"Hamiltonian constructed in {time.time() - start_time:.2f} seconds.")

    return Qobj(H_mat)

if __name__ == "__main__":
    start_time_total = time.time()

    all_states = generate_states(N_sites, N_max, N_total)
    orbits = get_orbits(all_states)
    infos, basis_states, representative_states = momentum_eigenstates(N_sites, N_max, orbits)

    for i, info in enumerate(infos):
        k, parity, N = info

        print()
        print(f"Calculating for k = {k}, parity = {parity}...")
        print("-" * 50)
        start_time_i = time.time()

        H = build_hamiltonian(N_sites, N_max, N_total, basis_states[i], representative_states[i], J, U, mu)

        print("Diagonalizing Hamiltonian...")
        start_time = time.time()
        eigenvals = np.sort(H.eigenenergies()) / N_total
        print(f"{len(eigenvals)} eigenvalues calculated in {time.time() - start_time:.2f} seconds.")

        print("-" * 50)
        print(f"Total time for k = {k}, parity = {parity}: {time.time() - start_time_i:.2f} seconds.")

        np.savetxt(f"output k={k}, parity={parity}.csv", eigenvals)
    
    print(f"Total time for all calculations: {time.time() - start_time_total:.2f} seconds.")