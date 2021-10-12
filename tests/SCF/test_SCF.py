import pytest
import SCF
import pickle
import numpy as np


def test_calc_nuclear_repulsion_energy(mol_h2o):
    assert SCF.calc_nuclear_repulsion_energy(mol_h2o) == 8.00236706181077, \
        "Nuclear Repulsion Energy Test (H2O) Failed"


def test_calc_hcore_matrix(mol_h2o):
    Suv = pickle.load(open("suv.pkl", "rb"))
    Tuv = pickle.load(open("tuv.pkl", "rb"))
    Vuv = pickle.load(open("vuv.pkl", "rb"))
    eri = pickle.load(open("eri.pkl", "rb"))
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    assert SCF.calc_hcore_matrix(Tuv, Vuv)[0, 0] == \
        pytest.approx(-32.57739541261037) and \
        SCF.calc_hcore_matrix(Tuv, Vuv)[3, 4] == 0.0 and \
        SCF.calc_hcore_matrix(Tuv, Vuv)[4, 3] == 0.0, \
        "Initial Hcore Matrix Test (H2O) Failed"


def test_calc_initial_density(mol_h2o):
    assert np.sum(SCF.calc_initial_density(mol_h2o)) == 0, \
        "Initial Density matrix Test (H2O) Failed"


def test_calc_fock_matrix(mol_h2o):
    Suv = pickle.load(open("suv.pkl", "rb"))
    Tuv = pickle.load(open("tuv.pkl", "rb"))
    Vuv = pickle.load(open("vuv.pkl", "rb"))
    eri = pickle.load(open("eri.pkl", "rb"))
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    Duv = SCF.calc_initial_density(mol_h2o)
    assert SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)[0, 0] == \
        pytest.approx(-32.57739541261037) and \
        SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)[2, 5] == \
        pytest.approx(-1.6751501447185015) and \
        SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)[5, 2] == \
        pytest.approx(-1.6751501447185015), \
        "Fock Matrix Test (H2O) Failed"


def test_solve_Roothan_equations(mol_h2o):
    Suv = pickle.load(open("suv.pkl", "rb"))
    Tuv = pickle.load(open("tuv.pkl", "rb"))
    Vuv = pickle.load(open("vuv.pkl", "rb"))
    eri = pickle.load(open("eri.pkl", "rb"))
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    Duv = SCF.calc_initial_density(mol_h2o)
    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    mo_e, mo_c = SCF.solve_Roothan_equations(Fuv, Suv)
    assert abs(mo_e) == pytest.approx([32.57830292,  8.08153571,  7.55008599,
                                       7.36396923,   7.34714487,  4.00229867,
                                       3.98111115]) \
        and abs(mo_c[0, :]) == pytest.approx([1.00154358e+00,  2.33624458e-01,
                                              4.97111543e-16, 8.56842145e-02,
                                              2.02299681e-29,  4.82226067e-02,
                                              4.99600361e-16]), \
        "Solve Roothan Equations Test (H2O) Failed"


def test_form_density_matrix(mol_h2o):
    Suv = pickle.load(open("suv.pkl", "rb"))
    Tuv = pickle.load(open("tuv.pkl", "rb"))
    Vuv = pickle.load(open("vuv.pkl", "rb"))
    eri = pickle.load(open("eri.pkl", "rb"))
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    Duv = SCF.calc_initial_density(mol_h2o)
    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    mo_e, mo_c = SCF.solve_Roothan_equations(Fuv, Suv)
    assert SCF.form_density_matrix(mol_h2o, mo_c)[0, 0] == \
        pytest.approx(2.130023428655504) and \
        SCF.form_density_matrix(mol_h2o, mo_c)[2, 5] == \
        pytest.approx(-0.29226330209653156) and \
        SCF.form_density_matrix(mol_h2o, mo_c)[5, 2] == \
        pytest.approx(-0.29226330209653156), \
        "Form Density Matrix Test (H2O) Failed"


def test_calc_tot_energy(mol_h2o):
    Suv = pickle.load(open("suv.pkl", "rb"))
    Tuv = pickle.load(open("tuv.pkl", "rb"))
    Vuv = pickle.load(open("vuv.pkl", "rb"))
    eri = pickle.load(open("eri.pkl", "rb"))
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    Duv = SCF.calc_initial_density(mol_h2o)
    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    Enuc = SCF.calc_nuclear_repulsion_energy(mol_h2o)
    assert SCF.calc_tot_energy(Fuv, Huv, Duv, Enuc) == \
        pytest.approx(8.0023670618), \
        "Total Energy Test (H2O) Failed"
