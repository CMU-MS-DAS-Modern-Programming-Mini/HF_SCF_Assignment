import pytest
import SCF
import numpy as np
from pyscf import gto  # PySCF is a quantum chemistry python module


def test_all_functions():
    # Input Data
    mol_h2o = gto.M(unit="Bohr",
                    atom="O 0.000000000000  -0.143225816552   0.000000000000;"
                    + "H 1.638036840407   1.136548822547  -0.000000000000;"
                    + "H -1.638036840407   1.136548822547  -0.000000000000",
                    basis='STO-3g')
    mol_h2o.build()

    E_conv_threshold = 1.0E-10
    D_conv_threshold = 1.0E-8
    max_iterations = 1000

    # get the integrals
    Suv = mol_h2o.intor('int1e_ovlp')  # Overlap Integrals
    Tuv = mol_h2o.intor('int1e_kin')  # Kinetic Energy 1 electron integrals
    Vuv = mol_h2o.intor('int1e_nuc')  # Nuclear Repulsion 1 electron integrals
    eri = mol_h2o.intor("int2e")  # Electron Repulsion 2 electron integrals

    Enuc = SCF.calc_nuclear_repulsion_energy(mol_h2o)
    assert Enuc == 8.00236706181077, \
        "Nuclear Repulsion Energy Test (H2O) Failed"

    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    assert np.isclose(Huv[0, 0], -32.57739541261037), "Hcore matrix fail"
    assert Huv[3, 4] == 0.0, "Hcore matrix fail"
    assert Huv[4, 3] == 0.0, "Hcore matrix fail"

    Duv = SCF.calc_initial_density(mol_h2o)
    for i in range(Duv.shape[0]):
        for j in range(Duv.shape[1]):
            assert Duv[i, j] == 0.0, "initial density matrix fail"

    Etot = 0.0
    for it in range(max_iterations):

        Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)

        mo_e, mo_c = SCF.solve_Roothan_equations(Fuv, Suv)

        Etot_new = SCF.calc_tot_energy(Fuv, Huv, Duv, Enuc)

        Duv_new = SCF.form_density_matrix(mol_h2o, mo_c)

        if (it == 0):
            assert np.isclose(Fuv[0, 0], -32.57739541261037), \
                "fock matrix iteration 1 fail"
            assert np.isclose(Fuv[2, 5], -1.6751501447185015), \
                "fock matrix iteration 1 fail"
            assert np.isclose(Fuv[5, 2], -1.6751501447185015), \
                "fock matrix iteration 1 fail"
            mol_e_a = [-32.57830292, -8.08153571, -7.55008599, -7.36396923]
            mol_e_b = [-7.34714487, -4.00229867, -3.98111115]
            mol_e = mol_e_a + mol_e_b
            for i in range(len(mol_e)):
                assert np.isclose(mo_e[i], mol_e[i]), \
                    "Roothan equation fail mol_e"
            assert np.isclose(Etot_new, 8.0023670618), "etot iteration 1 error"
            assert np.isclose(Duv_new[0, 0], 2.130023428655504), \
                "density matrix iteration 1 fail"
            assert np.isclose(Duv_new[2, 5], -0.29226330209653156), \
                "density matrix iteration 1 fail"
            assert np.isclose(Duv_new[5, 2], -0.29226330209653156), \
                "density matrix iteration 1 fail"
        if (it == 1):
            assert np.isclose(Fuv[0, 0], -18.81326949992384), \
                "fock matrix iteration 2 fail"
            assert np.isclose(Fuv[2, 5], -0.1708886336992761), \
                "fock matrix iteration 2 fail"
            assert np.isclose(Fuv[5, 2], -0.1708886336992761), \
                "fock matrix iteration 2 fail"
            assert np.isclose(Etot_new, -73.2857964211), \
                "etot iteration 2 error"

        dEtot = abs(Etot_new - Etot)
        dDuv = (((Duv_new - Duv)**2).sum())**(1.0/2.0)

        if dEtot < E_conv_threshold and dDuv < D_conv_threshold:
            print("Final Energy = {:.10f}".format(Etot_new))
            break

        print("Etot = {:.10f} dEtot = {:.10f} dDuv = {:.10f}".format(Etot_new,
                                                                     dEtot,
                                                                     dDuv))
        Duv = Duv_new.copy()
        Etot = Etot_new

    assert np.isclose(Etot, -74.9420799282), "final etot fails"
