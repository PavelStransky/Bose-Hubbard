import numpy as np
from itertools import product
from alive_progress import alive_it
from qutip import *
from multiprocessing import Pool
import time

# -----------------------------
# Parameters
# -----------------------------
N_sites = 3        # number of lattice sites
N_max = 50        # max bosons per site
N_total = 50      # fixed total boson number
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

    print(f"Constructing momentum eigenstates...")
    start_time = time.time()

    for orbit in alive_it(orbits):
        for i, info in enumerate(infos):
            k, parity, num = info

            basis_state = []
            for r, state in enumerate(orbit[0]):
                phase = np.exp(-2j * np.pi * k * r / N_sites)
                basis_state.append((phase, state))

            if len(orbit) > 1:
                basis_state_parity = []
                for r, state in enumerate(orbit[1]):
                    phase = np.exp(-2j * np.pi * k * r / N_sites)
                    basis_state_parity.append((phase, state))

                if parity == 0:
                    basis_states[i].append(basis_state)
                    basis_states[i].append(basis_state_parity)
                    info[2] += 2

                elif parity == -1:  # even parity
                    basis_state_parity = [(-phase, state) for phase, state in basis_state_parity]
                    basis_state.extend(basis_state_parity)
                    basis_states[i].append(basis_state)
                    info[2] += 1

                elif parity == 1:  # odd parity
                    basis_state.extend(basis_state_parity)
                    basis_states[i].append(basis_state)
                    info[2] += 1
            
            elif parity >= 0:
                basis_states[i].append(basis_state)
                info[2] += 1

    total = sum(info[2] for info in infos)
    print(f"Constructed {total} basis states in {time.time() - start_time:.2f} seconds.")

    return infos, basis_states
    
def remove_boson(state, site):
    new_state = list(state)
    new_state[site] -= 1
    return tuple(new_state)

def build_hamiltonian_block(N_sites, N_max, N_total, basis_states, J, U, mu, l):
    dim = len(basis_states)
    H_mat = np.zeros((dim, dim), dtype=complex)

    k = (l + 1) % N_sites

    for i1 in range(dim):
        state1 = basis_states[i1]

        for i2 in range(i1 + 1, dim):
            state2 = basis_states[i2]

            bra = [(phase.conj() * np.sqrt(s[l]), remove_boson(s, l)) for phase, s in state1]
            ket = [(phase * np.sqrt(s[k]), remove_boson(s, k)) for phase, s in state2]

            norm = 1 / np.sqrt(len(bra) * len(ket))

            for phase_bra, state_bra in bra:
                for phase_ket, state_ket in ket:
                    if state_bra == state_ket:
                        d = -norm * phase_bra * phase_ket * J
                        H_mat[i1, i2] += d
                        H_mat[i2, i1] += d.conjugate()  # ensure Hermitian

            bra = [(phase.conj() * np.sqrt(s[k]), remove_boson(s, k)) for phase, s in state1]
            ket = [(phase * np.sqrt(s[l]), remove_boson(s, l)) for phase, s in state2]

            norm = 1 / np.sqrt(len(bra) * len(ket))

            for phase_bra, state_bra in bra:
                for phase_ket, state_ket in ket:
                    if state_bra == state_ket:
                        d = -norm * phase_bra * phase_ket * J
                        H_mat[i1, i2] += d
                        H_mat[i2, i1] += d.conjugate()  # ensure Hermitian

    return H_mat

def build_hamiltonian(N_sites, N_max, N_total, basis_states, J, U, mu):
    """ Build the Bose-Hubbard Hamiltonian in the reduced momentum basis_states. """

    print("Building Hamiltonian in reduced momentum basis_states...")
    start_time = time.time()

    with Pool() as pool:
        results = pool.starmap(build_hamiltonian_block, [(N_sites, N_max, N_total, basis_states, J, U, mu, l) for l in range(N_sites)])

    # Merge all blocks
    dim = len(basis_states)
    H_mat = np.zeros((dim, dim), dtype=complex)

    # Add on-site interaction and chemical potential (diagonal)
    for i, state in enumerate(basis_states):
        energy = 0.0
        for n in state[0][1]:
            energy += (U / N_total) * n * (n - 1) - mu * n
        H_mat[i, i] += energy

    for matrices in results:
        H_mat += matrices

    print(f"Hamiltonian constructed in {time.time() - start_time:.2f} seconds.")

    return Qobj(H_mat)

if __name__ == "__main__":
    start_time_total = time.time()

    all_states = generate_states(N_sites, N_max, N_total)
    orbits = get_orbits(all_states)
    infos, basis_states = momentum_eigenstates(N_sites, N_max, orbits)

    for i, info in enumerate(infos):
        k, parity, N = info

        print()
        print(f"Calculating for k = {k}, parity = {parity}...")
        print("-" * 50)
        start_time_i = time.time()

        H = build_hamiltonian(N_sites, N_max, N_total, basis_states[i], J, U, mu)

        print("Diagonalizing Hamiltonian...")
        start_time = time.time()
        eigenvals = np.sort(H.eigenenergies()) / N_total
        print(f"{len(eigenvals)} eigenvalues calculated in {time.time() - start_time:.2f} seconds.")

        print("-" * 50)
        print(f"Total time for k = {k}, parity = {parity}: {time.time() - start_time_i:.2f} seconds.")

        np.savetxt(f"X output k={k}, parity={parity}.csv", eigenvals)
    
    print(f"Total time for all calculations: {time.time() - start_time_total:.2f} seconds.")