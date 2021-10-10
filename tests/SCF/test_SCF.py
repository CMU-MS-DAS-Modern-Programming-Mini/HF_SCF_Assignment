import pytest
import SCF
from pyscf import gto  # PySCF is a quantum chemistry python module
import numpy as np


def test_SCF():
    mol_h2o = gto.M(unit="Bohr",
                    atom="O 0.000000000000  -0.143225816552   0.000000000000;"
                    + "H 1.638036840407   1.136548822547  -0.000000000000;"
                    + "H -1.638036840407   1.136548822547  -0.000000000000",
                    basis='STO-3g')
    mol_h2o.build()

    # Convergence Criteria
    E_conv_threshold = 1.0E-10
    D_conv_threshold = 1.0E-8
    max_iterations = 1000

    # get the integrals
    Suv = mol_h2o.intor('int1e_ovlp')  # Overlap Integrals
    Tuv = mol_h2o.intor('int1e_kin')  # Kinetic Energy 1 electron integrals
    Vuv = mol_h2o.intor('int1e_nuc')  # Nuclear Repulsion 1 electron integrals
    eri = mol_h2o.intor("int2e")  # Electron Repulsion 2 electron integrals

    Enuc = SCF.calc_nuclear_repulsion_energy(mol_h2o)
    assert np.isclose(Enuc, 8.00236706181077), \
        "Nuclear Repulsion Energy Test (H2O) Failed"
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    assert np.isclose(Huv[0,0], -32.57739541261037) \
        and np.isclose(Huv[4,3], 0.0) \
            and np.isclose(Huv[3,4], 0.0), \
            "Calculating HCore Matrix Failed"
    
    Duv = SCF.calc_initial_density(mol_h2o)
    Etot = 0.0

    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    assert np.isclose(Fuv[0,0], -32.57739541261037) \
            and np.isclose(Fuv[5,2], -1.6751501447185015) \
                and np.isclose(Fuv[2,5], -1.6751501447185015), \
                    "Calculating Fock Matrix Failed"

    mo_e, mo_c = SCF.solve_Roothan_equations(Fuv, Suv)

    assert np.isclose(mo_e, [-32.57830292, -8.08153571 , -7.55008599, \
                -7.36396923 ,  -7.34714487,  -4.00229867 ,\
                    -3.98111115]).all(), \
          "Failure in solving Roothan equations, mol_e incorrect"

    """
    assert np.isclose(mo_c[0, :], [-1.00154358e+00 , 2.33624458e-01 , 4.97111543e-16, \
            -8.56842145e-02 , 2.02299681e-29 , 4.82226067e-02, \
                -4.99600361e-16]).all(),\
                "Failure in solving Roothan equations, mol_c value incorrect"
    """
    #we are not guaranteed to get the same values when solving for eigenvectors!

    Etot_new = SCF.calc_tot_energy(Fuv, Huv, Duv, Enuc)

    assert np.isclose(Etot_new, 8.0023670618), "Failure in calculaitng total energy."

    Duv_new = SCF.form_density_matrix(mol_h2o, mo_c)

    assert np.isclose(Duv_new[0,0], 2.130023428655504) and np.isclose(Duv_new[5,2], -0.29226330209653156)\
        and np.isclose(Duv_new[2,5], -.29226330209653156), "Failure in calculating new density matrix"

    return True

test_SCF()
