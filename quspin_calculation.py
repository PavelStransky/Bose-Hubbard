from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d

import numpy as np
from alive_progress import alive_it, alive_bar

import matplotlib.pyplot as plt
import multiprocessing

import time

PATH = "results/"

def entropy(basis, V, sub_sys_A):
    return float(basis.ent_entropy(V, sub_sys_A=sub_sys_A)['Sent_A'])

if __name__ == "__main__":
    L = 8          # number of sites
    N = 12         # number of boson excitations
    U = 1.0

    for J in np.arange(-1, 1, 0.05):
        iprs = []
        entanglement_entropies = []
        es = []
        infos = []

        for k in range(L):
            for parity in [1, -1]:
                if parity == -1 and k != 0 and 2*k != L:
                    continue

                print(f"Calculating for J={J}, k={k}, parity={parity}...")

                basis = boson_basis_1d(L, Nb=N, sps=N + 1, kblock=k, pblock=parity)
                # print(basis)

                hopping = [[-J, i, (i + 1) % L] for i in range(L)]  # periodic hopping
                hubbard = [[U / N, i, i] for i in range(L)]         # on-site interaction
                chemical = [[-U / N, i] for i in range(L)]          # chemical potential

                static = [["+-", hopping], ["-+", hopping], ["n", chemical], ["nn", hubbard]]
                H = hamiltonian(static, [], basis=basis, dtype=np.float64)

                start_time = time.time()
                E, V = H.eigh()
                es.append(E / N)

                print(f"{len(E)} eigenvalues calculated in {time.time() - start_time:.2f} seconds.")

                iprs.append(np.array([1 / sum(abs(v)**4 for v in vec) / len(vec) for vec in alive_it(V.T)]))

                entanglement_entropy = []
                with multiprocessing.Pool(processes=8) as pool, alive_bar(len(E)) as bar:
                    entropy_object = []

                    def update_entropy_object(_):            
                        bar()

                    for args in [(basis, V[i], list(range(L-1))) for i in range(len(E))]:
                        r = pool.apply_async(entropy, args=args, callback=update_entropy_object)
                        entropy_object.append(r)

                    pool.close()
                    pool.join()

                    entanglement_entropy = [r.get() for r in entropy_object]

                entanglement_entropies.append(np.array(entanglement_entropy))

                infos.append(f"{k}{"+" if parity == 1 else "-"}")

                np.savetxt(PATH + f"L={L} N={N} J={J:.2f} U={U} k={k} parity={parity}.csv", es[-1])
                np.savetxt(PATH + f"L={L} N={N} J={J:.2f} U={U} k={k} parity={parity} entropy.csv", entanglement_entropies[-1])
                np.savetxt(PATH + f"L={L} N={N} J={J:.2f} U={U} k={k} parity={parity} ipr.csv", iprs[-1])

        plt.figure(figsize=(10, 6))
        energies = []
        for energy in es:
            energies.extend(energy)
        
        plt.hist(energies, bins=100, density=True)
        plt.xlabel("Energy")
        plt.ylabel("Rho")
        plt.savefig(PATH + f"rho L={L} N={N} J={J} U={U}.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        for info, e, ipr in zip(infos, es, iprs):
            plt.scatter(e, ipr, s=1, label=info)
        plt.xlabel("Energy")
        plt.ylabel("Inverse Participation Ratio (IPR)")
        plt.title(f"IPR for L={L}, N={N}, J={J}, U={U}")
        plt.legend()
        plt.savefig(PATH + f"ipr L={L} N={N} J={J} U={U}.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        for info, e, entanglement_entropy in zip(infos, es, entanglement_entropies):
            plt.scatter(e, entanglement_entropy, s=1, label=info)
        plt.xlabel("Energy")
        plt.ylabel("Entanglement Entropy")
        plt.title(f"Entanglement Entropy for L={L}, N={N}, J={J}, U={U}")
        plt.legend()
        plt.savefig(PATH + f"entropy L={L} N={N} J={J} U={U} k={k}, parity={parity}.png")
        plt.close()

        print("-" * 100)