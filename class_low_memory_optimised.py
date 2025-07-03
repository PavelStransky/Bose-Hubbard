import numpy as np
from itertools import product
from alive_progress import alive_it
from qutip import *
from multiprocessing import Pool
import time

class BoseHubbard:
    def __init__(self, N_sites, N_max, N_total, J, U, mu, translation_sector=0, parity_sector=1):
        """ Initialize the Bose-Hubbard model parameters. """
        self.N_sites = N_sites
        self.N_max = N_max
        self.N_total = N_total
        self.J = J
        self.U = U
        self.mu = mu

        self.translation_sector = translation_sector
        self.parity_sector = parity_sector

    def create_orbits(self):
        """ Find translation orbits and their parity invariants. """
    
        def translate_state(state, shift):
            """ Return the translated Fock state tuple by shift. """
            return tuple(state[(i - shift) % self.N_sites] for i in range(self.N_sites))

        def parity_reflect_state(state):
            """Return the parity-reflected Fock state tuple."""
            return tuple(reversed(state))

        print(f"Constructing translation orbits...")
        single_states = [tuple(s) for s in product(range(self.N_max + 1), repeat=self.N_sites) if sum(s) == self.N_total]

        self.orbits = []

        covered = set()
        for state in alive_it(single_states):
            if state in covered:
                continue

            orbit = {translate_state(state, r) for r in range(self.N_sites)}
            parity_orbit = {parity_reflect_state(state) for state in orbit}

            parity_invariant = orbit == parity_orbit

            covered.update(orbit)
            covered.update(parity_orbit)

            # We require the same order for both state and parity state
            orbit = sorted(orbit)
            parity_orbit = [parity_reflect_state(state) for state in orbit]

            if parity_invariant:
                orbit = [orbit]
            else:
                orbit = [orbit, parity_orbit]

            self.orbits.append(orbit)

    def create_momentum_eigenstates(self):   
        """ Construct the momentum and parity eigenstates."""

        def normalise_state(state):
            """ Normalize the state tuple. """
            norm = sum(abs(phase)**2 for phase, _ in state) ** 0.5
            return [(phase / norm, s) for phase, s in state]

        print(f"Constructing normalised momentum eigenstates...")
        self.basis_states = []

        for orbit in alive_it(self.orbits):
            basis_state = []
            for r, state in enumerate(orbit[0]):
                phase = np.exp(-2j * np.pi * self.translation_sector * r / self.N_sites)
                basis_state.append((phase, state))

            if len(orbit) > 1:
                basis_state_parity = []
                for r, state in enumerate(orbit[1]):
                    phase = np.exp(-2j * np.pi * self.translation_sector * r / self.N_sites)
                    basis_state_parity.append((phase, state))

                if self.parity_sector == 0:
                    self.basis_states.append(normalise_state(basis_state))
                    self.basis_states.append(normalise_state(basis_state_parity))

                elif self.parity_sector == -1:  # even parity
                    basis_state_parity = [(-phase, state) for phase, state in basis_state_parity]
                    basis_state.extend(basis_state_parity)
                    self.basis_states.append(normalise_state(basis_state))

                elif self.parity_sector == 1:  # odd parity
                    basis_state.extend(basis_state_parity)
                    self.basis_states.append(normalise_state(basis_state))
            
            elif self.parity_sector >= 0:
                self.basis_states.append(normalise_state(basis_state))
    
    def build_hamiltonian_block(self, l):
        def remove_boson(state, site):
            new_state = list(state)
            new_state[site] -= 1
            return tuple(new_state)

        dim = len(self.basis_states)
        H_mat = np.zeros((dim, dim), dtype=complex)

        k = (l + 1) % self.N_sites

        bras = []
        kets = []

        for state in self.basis_states:
            bras.append([(phase.conj() * np.sqrt(s[l]), remove_boson(s, l)) for phase, s in state])
            kets.append([(phase * np.sqrt(s[k]), remove_boson(s, k)) for phase, s in state])

        for i1, bra in enumerate(bras):
            for i2, ket in enumerate(kets):
                for phase_bra, state_bra in bra:
                    for phase_ket, state_ket in ket:
                        if state_bra == state_ket:
                            d = phase_bra * phase_ket * self.J
                            H_mat[i1, i2] += d

        bras = []
        kets = []
        
        for state in self.basis_states:
            bras.append([(phase.conj() * np.sqrt(s[k]), remove_boson(s, k)) for phase, s in state])
            kets.append([(phase * np.sqrt(s[l]), remove_boson(s, l)) for phase, s in state])    

        for i1, bra in enumerate(bras):
            for i2, ket in enumerate(kets):
                for phase_bra, state_bra in bra:
                    for phase_ket, state_ket in ket:
                        if state_bra == state_ket:
                            d = phase_bra * phase_ket * self.J
                            H_mat[i1, i2] += d

        return H_mat

    def build_hamiltonian(self):
        """ Build the Bose-Hubbard Hamiltonian in the reduced momentum basis_states. """

        self.create_orbits()
        self.create_momentum_eigenstates()

        print(f"Constructing Hamiltonian for {len(self.basis_states)} basis states in {N_sites} threads...")
        start_time = time.time()

        with Pool() as pool:
            results = pool.map(self.build_hamiltonian_block, range(N_sites))

        print("Merging blocks and adding diagonal terms...")

        # Merge all blocks
        dim = len(self.basis_states)
        H_mat = np.zeros((dim, dim), dtype=complex)

        # Add on-site interaction and chemical potential (diagonal)
        for i, state in enumerate(self.basis_states):
            energy = 0.0
            for n in state[0][1]:
                energy += (self.U / self.N_total) * n * (n - 1) - self.mu * n
            H_mat[i, i] += energy

        for matrices in results:
            H_mat += matrices

        print(f"Hamiltonian constructed in {time.time() - start_time:.2f} seconds.")

        return Qobj(H_mat)

    def eigenstates(self):
        """ Calculate the eigenstates of the Hamiltonian. """
        
        H = self.build_hamiltonian()

        print("Calculating eigenstates...")
        start_time = time.time()

        eigenvals, eigenkets = H.eigenstates()
        eigenvals /= self.N_total

        print(f"Eigenvalues calculated in {time.time() - start_time:.2f} seconds.")

        return eigenvals

if __name__ == "__main__":
    # -----------------------------
    # Parameters
    # -----------------------------
    N_sites = 3        # number of lattice sites
    N_max = 50        # max bosons per site
    N_total = 50      # fixed total boson number
    translational_sector = 1       # momentum sector (integer from 0 to N_sites-1)
    parity_sector = 0

    J = -1.0            # hopping amplitude
    U = -5.0            # on-site interaction
    mu = 0.0           # chemical potential

    start_time_total = time.time()
    bh = BoseHubbard(N_sites, N_max, N_total, J, U, mu, translation_sector=translational_sector, parity_sector=parity_sector)
    eigenvals = bh.eigenstates()

    print(f"{len(eigenvals)} eigenvalues calculated in total time of {time.time() - start_time_total:.2f} seconds.")

    np.savetxt(f"Z output k={translational_sector}, parity={parity_sector}.csv", eigenvals)
    