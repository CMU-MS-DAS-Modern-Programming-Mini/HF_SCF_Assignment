import pytest
import SCF


def test_calc_nuclear_repulsion_energy(mol_h2o):
    assert True
#    assert SCF.calc_nuclear_repulsion_energy(mol_h2o) == 8.00236706181077,\
#        "Nuclear Repulsion Energy Test (H2O) Failed"
import pytest
import SCF
import main
import pickle

# Overlaps the Integrals
Suv = pickle.load(open("suv.pkl", "rb"))
# Kinetic Energy 1 electron integrals
Tuv = pickle.load(open("tuv.pkl", "rb"))
# Nuclear Repulsion 1 electron integrals
Vuv = pickle.load(open("vuv.pkl", "rb"))
# Electron Repulsion 2 electron integrals
eri = pickle.load(open("eri.pkl", "rb"))

def test_calc_nuclear_repulsion_energy(mol_h2o):  # checks nuclear repulsion energy and gives out error messaage if incorrect
    assert SCF.calc_nuclear_repulsion_energy(mol_h2o) == 8.00236706181077,"Nuclear Repulsion Energy Test (H2O) Failed"
    
def test_calc_initial_density(mol_h2o): 
    """
    Tests that the initial density returns a zero matrix
    and tests dimensions
    """

    Duv = SCF.calc_initial_density(mol_h2o)
    assert Duv.sum() == 0.0
    assert Duv.shape == (mol_h2o.nao,mol_h2o.nao)

def test_calc_hcore_matrix():  # checks Hcore matrix
    Tuv = pickle.load(open("tuv.pkl", "rb"))
    Vuv = pickle.load(open("vuv.pkl", "rb"))
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    assert Huv[0, 0] == -32.57739541261037
    assert Huv[3, 4] == 0.0


def test_calc_fock_matrix(mol_h2o):  # checks fock matrix
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    Duv = SCF.calc_initial_density(mol_h2o)
    Fuv= SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    assert Fuv[0,0] == -32.57739541261037
    assert Fuv[2,5] == pytest.approx(-1.6751501447185015)  # needed because the answers are not exactly equal
    assert Fuv[5,2] == pytest.approx(-1.6751501447185015)

def test_solve_Roothan_equations(mol_h2o):  # checks Roothan euations (eigen vectors and eigen values)
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    Duv = SCF.calc_initial_density(mol_h2o)
    Fuv= SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    mo_e, mo_c = SCF.solve_Roothan_equations(Fuv, Suv)
    assert mo_e[0] == pytest.approx(-32.5783029) # checking one value from the array

def test_calc_total_energy(mol_h2o):  # checks total energy
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    Duv = SCF.calc_initial_density(mol_h2o)
    Fuv= SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    Enuc = SCF.calc_nuclear_repulsion_energy(mol_h2o)
    Etot_new = SCF.calc_total_energy(Fuv, Huv, Duv, Enuc)
    assert Etot_new ==  pytest.approx(8.0023670618)

def test_form_density_matrix(mol_h2o):  # checks Duv new over 1 iteration
    
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    Duv = SCF.calc_initial_density(mol_h2o)
    Fuv= SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    mo_e, mo_c = SCF.solve_Roothan_equations(Fuv, Suv)
    Duv_new = SCF.form_density_matrix(mol_h2o, mo_c)
    assert Duv_new[0,0] == 2.130023428655504
    assert Duv_new[2,5] == pytest.approx(-0.29226330209653156) 
    assert Duv_new[5,2] == pytest.approx(-0.29226330209653156) 