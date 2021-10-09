import pytest
import SCF


def test_calc_nuclear_repulsion_energy(mol_h2o):
    assert True
    assert SCF.calc_nuclear_repulsion_energy(mol_h2o) == pytest.approx(8.00236706181077),\
        "Nuclear Repulsion Energy Test (H2O) Failed"

def test_calc_initial_density(mol_h2o):
    Duv = SCF.calc_initial_density(mol_h2o)
    assert Duv.sum() == 0
    assert Duv.shape == (mol_h2o.nao, mol_h2o.nao)

def test_calc_hcore_matrix(mol_h2o):
    Tuv = mol_h2o.intor("int1e_kin")
    Vuv = mol_h2o.intor("int1e_nuc")
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)

    assert Huv[0][0] == pytest.approx(-32.57739541261037), \ 
        "Calculate Hcore Matrix Test (H2O) Failed"

    assert Huv(3, 4) == 0.0, \ 
        "Calculate Hcore Matrix Test (H2O) Failed"
    
    assert Huv(4, 3) == 0.0, \ 
        "Calculate Hcore Matrix Test (H2O) Failed"

def test_calc_fock_matrix(mol_h2o):
    Duv = SCF.calc_initial_density(mol_h2o)
    Tuv = mol_h2o.intor("int1e_kin")
    Vuv = mol_h2o.intor("int1e_nuc")
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    eri = mol_h2o.intor("int2e")
    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)

    assert Fuv[0][0] == pytest.approx(-18.81326949992384), \
        "Calculate Fock Matrix Test (H2O) Failed"

    assert Fuv[2][5] == pytest.approx(-0.1708886336992761), \
        "Calculate Fock Matrix Test (H2O) Failed"
    
    assert Fuv[5][2] == pytest.approx(-0.1708886336992761), \
        "Calculate Fock Matrix Test (H2O) Failed"

def test_solve_Roothan_equations(mol_h2o):
    Duv = SCF.calc_initial_density(mol_h2o)
    Tuv = mol_h2o.intor("int1e_kin")
    Vuv = mol_h2o.intor("int1e_nuc")
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    eri = mol_h2o.intor("int2e")
    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    Suv = mol_h2o.intor("int1e_ovlp")

    mol_e, mol_c = SCF.solve_Roothan_equations(Fuv, Suv)
    assert mol_e == pytest.approx([-32.57830292, -8.08153571, -7.55008599,  
                     -7.36396923, -7.34714487, -4.00229867,  
                     -3.98111115]), \
        "Solve Roothan Equations Test (H2O) Failed"

    assert mol_c[0,:] == pytest.approx([-1.00154358e+00, 2.33624458e-01, 4.97111543e-16,
                          -8.56842145e-02, 2.02299681e-29, 4.82226067e-02,
                          -4.99600361e-16]), \
        "Solve Roothan Equations Test (H2O) Failed"   

def test_form_density_matrix(mol_h2o):

    Duv = SCF.calc_initial_density(mol_h2o)
    Tuv = mol_h2o.intor("int1e_kin")
    Vuv = mol_h2o.intor("int1e_nuc")
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    eri = mol_h2o.intor("int2e")
    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    Suv = mol_h2o.intor("int1e_ovlp")

    mol_e, mol_c = SCF.solve_Roothan_equations(Fuv, Suv)

    Duv_new = SCF.form_density_matrix(mol_h2o)

    assert Duv_new[0][0] == pytest.approx(2.130023428655504), \
        "Form Density Matrix Test (H2O) Failed"
    assert Duv_new[2][5] == pytest.approx(-0.29226330209653156), \
        "Form Density Matrix Test (H2O) Failed"
    assert Duv_new[5][2] == pytest.approx(-0.29226330209653156), \
        "Form Density Matrix Test (H2O) Failed"

def test_calc_tot_energy(mol_h2o):
    Duv = SCF.calc_initial_density(mol_h2o)
    Tuv = mol_h2o.intor("int1e_kin")
    Vuv = mol_h2o.intor("int1e_nuc")
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    eri = mol_h2o.intor("int2e")
    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    Suv = mol_h2o.intor("int1e_ovlp")

    Etot = SCF.calc_tot_energy(Fuv, Huv_, Duv_, Enuc_)
    assert Etot == pytest.approx(8.0023670618),\
        "Total Energy Test Failed"