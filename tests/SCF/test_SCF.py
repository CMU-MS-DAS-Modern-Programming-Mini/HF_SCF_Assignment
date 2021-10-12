from tests.conftest import mol_h2o
import pytest
import SCF


def test_calc_nuclear_repulsion_energy(mol_h2o):
    # test and assert repulsion calculation matches
    assert SCF.calc_nuclear_repulsion_energy(mol_h2o) == 8.00236706181077,\
        "Nuclear Repulsion Energy Test (H2O) Failed"


def test_calc_initial_density(mol_h2o):
    # test Duv matrix formation and right dimensions
    Duv = SCF.calc_initial_density(mol_h2o)

    # assert function matches
    assert Duv.sum() == 0
    assert Duv.shape == (mol_h2o.nao, mol_h2o.nao)


def test_calc_hcore_matrix(mol_h2o):
    # inputs: Tuv, Vuv
    Tuv = mol_h2o.intor("int1e_kin")
    Vuv = mol_h2o.intor("int1e_nuc")

    # test Huv calculation
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)

    # assert function matches
    assert Huv[0][0] == -32.57739541261037,\
        "H_Core Test Failed"

    assert Huv[3][4] == 0.0,\
        "H_Core Test Failed"


def test_calc_fock_matrix(mol_h2o):
    # inputs: Huv, eri, Duv
    Duv = SCF.calc_initial_density(mol_h2o)
    Tuv = mol_h2o.intor("int1e_kin")
    Vuv = mol_h2o.intor("int1e_nuc")
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    eri = mol_h2o.intor("int2e")

    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)

    # assert function matches
    assert Fuv[0][0] == -32.57739541261037,\
        "Fock Matrix Test Failed"

    assert Fuv[2][5] == pytest.approx(-1.6751501447185015),\
        "Fock Matrix Test Failed"


def test_solve_Roothan_equations(mol_h2o):
    # inputs: Fuv and Suv
    Suv = mol_h2o.intor("int1e_ovlp")
    Tuv = mol_h2o.intor("int1e_kin")
    Vuv = mol_h2o.intor("int1e_nuc")
    eri = mol_h2o.intor("int2e")
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    Duv = SCF.calc_initial_density(mol_h2o)
    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)

    # test Roothan equation calculation
    mo_e, mo_c = SCF.solve_Roothan_equations(Fuv, Suv)

    # assert statements
    assert mo_e == pytest.approx([-32.5783029, -8.08153571, -7.55008599,
                                  -7.36396923,  -7.34714487, -4.00229867,
                                  -3.98111115]),\
        "Roothan Equation Test Failed"

    assert mo_c[0, :] == pytest.approx([-1.00154358e+00,  -2.33624458e-01,
                                        4.97111543e-16, -8.56842145e-02,
                                        2.02299681e-29,  4.82226067e-02,
                                        -4.99600361e-16]),\
        "Roothan Equation Test Failed"


def test_form_density_matrix(mol_h2o):
    # input: mol_c
    Suv = mol_h2o.intor("int1e_ovlp")
    Duv = SCF.calc_initial_density(mol_h2o)
    Tuv = mol_h2o.intor("int1e_kin")
    Vuv = mol_h2o.intor("int1e_nuc")
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    eri = mol_h2o.intor("int2e")
    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    mol_e, mol_c = SCF.solve_Roothan_equations(Fuv, Suv)

    Duv_new = SCF.form_density_matrix(mol_h2o, mol_c)

    # assert statements
    assert Duv_new[0][0] == pytest.approx(2.130023428655504),\
        "Density Matrix Test Failed"

    assert Duv_new[2][5] == pytest.approx(-0.29226330209653156),\
        "Density Matrix Test Failed"


def test_calc_tot_energy(mol_h2o):
    # inputs: Fuv, Huv, Duv, Enuc
    Duv = SCF.calc_initial_density(mol_h2o)
    Enuc = SCF.calc_nuclear_repulsion_energy(mol_h2o)
    Tuv = mol_h2o.intor("int1e_kin")
    Vuv = mol_h2o.intor("int1e_nuc")
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    eri = mol_h2o.intor("int2e")
    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)

    # test total energy calculation
    Etot = SCF.calc_tot_energy(Fuv, Huv, Duv, Enuc)

    # assert statements match
    assert Etot == pytest.approx(8.0023670618),\
        "Total Energy Test Failed"
