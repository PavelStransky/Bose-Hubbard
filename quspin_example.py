from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d

import numpy as np
from alive_progress import alive_it

import matplotlib.pyplot as plt

import time

L = 5           # number of sites
N = 30         # number of boson excitations

J = 1.0
U = -5.0

k = 0           # momentum sector (integer from 0 to L-1)
parity = 1      # parity sector (integer: -1, 1)

basis = boson_basis_1d(L, Nb=N, sps=N + 1, kblock=k, pblock=parity)
print(basis)

hopping = [[-J, i, (i + 1) % L] for i in range(L)]  # periodic hopping
hubbard = [[U / N, i, i] for i in range(L)]         # on-site interaction
chemical = [[-U / N, i] for i in range(L)]          # chemical potential

static = [["+-", hopping], ["-+", hopping], ["n", chemical], ["nn", hubbard]]
H = hamiltonian(static, [], basis=basis, dtype=np.float64)

start_time = time.time()
E, V = H.eigh()

E /= N

print(f"{len(E)} eigenvalues calculated in {time.time() - start_time:.2f} seconds.")

plt.hist(E, bins=100, density=True)
plt.xlabel("Energy")
plt.ylabel("Rho")
plt.show()

ipr = [1 / sum(abs(v)**4 for v in vec) / len(vec) for vec in alive_it(V.T)]
plt.scatter(E, ipr, s=1)
plt.xlabel("Energy")
plt.ylabel("Inverse Participation Ratio (IPR)")
plt.show()

entanglement_entropy = []
for a in alive_it(range(len(E))):
    entanglement_entropy.append(float(basis.ent_entropy(V[a], sub_sys_A=[1,2,3])['Sent_A']))

plt.scatter(E, entanglement_entropy, s=1)
plt.show()

np.savetxt(f"R output k={k}, parity={parity}.csv", E / N)