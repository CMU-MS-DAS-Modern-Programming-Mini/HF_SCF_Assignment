import pytest
from pytest import approx
import SCF


def test_calc_nuclear_repulsion_energy(mol_h2o):
    assert True
    assert SCF.calc_nuclear_repulsion_energy(mol_h2o) == 8.00236706181077,\
        "Nuclear Repulsion Energy Test (H2O) Failed"


def test_calc_initial_density(mol_h2o):
    """
    Tests that the initial density returns a zero matrix
    and tests dimensions
    """

    Duv = SCF.calc_initial_density(mol_h2o)
    assert Duv.sum() == 0.0
    assert Duv.shape == (mol_h2o.nao, mol_h2o.nao)


def test_calc_hcore_matrix(mol_h2o):
    Tuv = mol_h2o.intor('int1e_kin')  # Kinetic Energy 1 electron integrals
    Vuv = mol_h2o.intor('int1e_nuc')  # Nuclear Repulsion 1 electron integrals
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    assert Huv[0, 0] == -32.57739541261037
    assert Huv[3, 4] == Huv[4, 3]
    assert Huv[3, 4] == 0.0


def test_calc_fock_matrix(mol_h2o):
    Tuv = mol_h2o.intor('int1e_kin')  # Kinetic Energy 1 electron integrals
    Vuv = mol_h2o.intor('int1e_nuc')  # Nuclear Repulsion 1 electron integrals
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    eri = mol_h2o.intor("int2e")
    Duv = SCF.calc_initial_density(mol_h2o)

    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    assert Fuv[0, 0] == -32.57739541261037
    assert Fuv[2, 5] == Fuv[5, 2]
    assert Fuv[2, 5] == approx(-1.6751501447185015)


def test_solve_Roothan_equations(mol_h2o):
    Tuv = mol_h2o.intor('int1e_kin')  # Kinetic Energy 1 electron integrals
    Vuv = mol_h2o.intor('int1e_nuc')  # Nuclear Repulsion 1 electron integrals
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    eri = mol_h2o.intor("int2e")
    Duv = SCF.calc_initial_density(mol_h2o)
    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    Suv = mol_h2o.intor('int1e_ovlp')

    mol_e, mol_c = SCF.solve_Roothan_equations(Fuv, Suv)
    assert mol_e == approx([-32.57830292, -8.08153571, -7.55008599,
                            -7.36396923, -7.34714487, -4.00229867,
                            -3.98111115])
    assert mol_c[0, :] == approx([-1.00154358e+00, 2.33624458e-01,
                                  4.97111543e-16, -8.56842145e-02,
                                  2.02299681e-29, 4.82226067e-02,
                                  -4.99600361e-16])


def test_calc_tot_energy(mol_h2o):
    Enuc = SCF.calc_nuclear_repulsion_energy(mol_h2o)
    Tuv = mol_h2o.intor('int1e_kin')  # Kinetic Energy 1 electron integrals
    Vuv = mol_h2o.intor('int1e_nuc')  # Nuclear Repulsion 1 electron integrals
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    eri = mol_h2o.intor("int2e")
    Duv = SCF.calc_initial_density(mol_h2o)
    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    Suv = mol_h2o.intor('int1e_ovlp')
    mol_e, mol_c = SCF.solve_Roothan_equations(Fuv, Suv)

    Etot_new = SCF.calc_tot_energy(Fuv, Huv, Duv, Enuc)
    assert Etot_new == 8.0023670618


def test_form_density_matrix(mol_h2o):
    Enuc = SCF.calc_nuclear_repulsion_energy(mol_h2o)
    Tuv = mol_h2o.intor('int1e_kin')  # Kinetic Energy 1 electron integrals
    Vuv = mol_h2o.intor('int1e_nuc')  # Nuclear Repulsion 1 electron integrals
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    eri = mol_h2o.intor("int2e")
    Duv = SCF.calc_initial_density(mol_h2o)
    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    Suv = mol_h2o.intor('int1e_ovlp')
    mol_e, mol_c = SCF.solve_Roothan_equations(Fuv, Suv)
    Etot_new = SCF.calc_tot_energy(Fuv, Huv, Duv, Enuc)

    Duv_new = SCF.form_density_matrix(mol_h2o, mol_c)
    assert Duv_new[0, 0] == 2.130023428655504
    assert Duv_new[2, 5] == Duv_new[5, 2]
    assert Duv_new[5, 2] == -0.29226330209653156
