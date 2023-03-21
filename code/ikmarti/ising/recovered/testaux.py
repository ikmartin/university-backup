import ising
import spinspace
import numpy as np


def run_aux_experiment(G, A, aux, display=False):
    G.set_aux(A, aux)
    if display:
        print("aux array: ", aux)
        print()

    h, J = G.get_qvec(display=display)

    if display:
        print("\n------------------------\n")

    fail = G.check_qvec(display=display)
    energy = G.energy_q_score()
    # display
    if display:
        print("# of failures: ", fail)
        print("  avg. energy: ", energy)
        print("\n------------------------------\n------------------------------\n")

    return fail, energy


def XOR_demo(A):
    XOR = ising.IXOR(1)
    aux_space = spinspace.get_spin_space(2 ** XOR.get_N() * A)
    for aux in aux_space:
        fail = run_aux_experiment(XOR, A, aux, display=True)
        if fail == 0:
            break


def MUL_demo(A, n):
    # A: number of aux spins
    # n: number of samples
    MUL = ising.IMUL(2, 2, A)
    auxLen = 2 ** MUL.get_N() * A

    # create dictionaries to store fails and energies and auxes
    fails = {}
    energies = {}
    auxes = {}
    for i in range(n):
        aux = spinspace.randspin(auxLen)
        j = spinspace.spin2dec(aux)
        fail, energy = run_aux_experiment(MUL, A, aux, display=False)
        fails[j] = fail
        energies[j] = energy
        auxes[j] = aux
        if fail == 0:
            break

    minE_aux = auxes[min(energies, key=energies.get)]
    minF_aux = auxes[min(fails, key=fails.get)]
    print("MINIMUM ENERGY")
    run_aux_experiment(MUL, A, minE_aux, display=True)
    print("MINIMUM FAILURE NUMBER")
    run_aux_experiment(MUL, A, minF_aux, display=True)


if __name__ == "__main__":
    A = 1
    # XOR_demo(A)
    # MUL_demo(A, 5000)
    # XOR = ising.IXOR(1)
    # run_aux_experiment(XOR, 1, [1, 1, -1, 1], display=True)
    MUL = ising.IMUL(2, 2, A)
    auxLen = 2 ** MUL.get_N() * A
    MUL2 = MUL.circuit_from_qvec()
    for spin in MUL._in_spins:
        s1 = np.concatenate((spin, MUL.get_value(spin)[0 : MUL.M]))
        s2 = np.concatenate((spin, np.array([-1, -1, -1, -1])))
        print("H(s1) = {}  ,  {} = H(s2)".format(MUL2.get_ham(s1), MUL2.get_ham(s2)))
    MUL2.h
    MUL2.J
