from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d

import numpy as np
from alive_progress import alive_it, alive_bar

import matplotlib.pyplot as plt
import multiprocessing

import time

def entropy(basis, V, sub_sys_A):
    return float(basis.ent_entropy(V, sub_sys_A=sub_sys_A)['Sent_A'])


if __name__ == "__main__":
    L = 4           # number of sites
    N = 50         # number of boson excitations

    J = 0.2
    U = 1.0

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

    ipr = [1 / sum(abs(v)**4 for v in vec) / len(vec) for vec in alive_it(V.T)]

    entanglement_entropy = []
    with multiprocessing.Pool(processes=10) as pool, alive_bar(len(E)) as bar:
        entropy_object = []

        def update_entropy_object(_):            
            bar()

        for args in [(basis, V[i], [1, 2, 3]) for i in range(len(E))]:
            r = pool.apply_async(entropy, args=args, callback=update_entropy_object)
            entropy_object.append(r)

        pool.close()
        pool.join()

        entanglement_entropy = [r.get() for r in entropy_object]

    entanglement_entropy = np.array(entanglement_entropy)

    np.savetxt(f"L={L} N={N} J={J} U={U} k={k}, parity={parity}.csv", E / N)
    np.savetxt(f"L={L} N={N} J={J} U={U} k={k}, parity={parity} entropy.csv", entanglement_entropy)
    np.savetxt(f"L={L} N={N} J={J} U={U} k={k}, parity={parity} ipr.csv", ipr)

    plt.hist(E, bins=100, density=True)
    plt.xlabel("Energy")
    plt.ylabel("Rho")
    plt.show()

    plt.scatter(E, ipr, s=1)
    plt.xlabel("Energy")
    plt.ylabel("Inverse Participation Ratio (IPR)")
    plt.show()

    plt.scatter(E, entanglement_entropy, s=1)
    plt.show()