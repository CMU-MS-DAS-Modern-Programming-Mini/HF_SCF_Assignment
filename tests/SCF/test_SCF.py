from pytest import approx
import SCF
import pytest
from pyscf import gto  # PySCF is a quantum chemistry python module
import numpy as np
import sys
sys.path.append("/Users/mkx/Desktop/Courses/Python/HF_SCF_Assignment")
print(sys.path)
mol_h2o = gto.M(unit="Bohr",
                atom="O 0.000000000000  -0.143225816552   0.000000000000;"
                + "H 1.638036840407   1.136548822547  -0.000000000000;"
                + "H -1.638036840407   1.136548822547  -0.000000000000",
                basis='STO-3g')
mol_h2o.build()
Suv = mol_h2o.intor('int1e_ovlp')  # Overlap Integrals
Tuv = mol_h2o.intor('int1e_kin')  # Kinetic Energy 1 electron integrals
Vuv = mol_h2o.intor('int1e_nuc')  # Nuclear Repulsion 1 electron integrals
eri = mol_h2o.intor("int2e")  # Electron Repulsion 2 electron integrals
Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
Duv = SCF.calc_initial_density(mol_h2o)

# @pytest.fixture()
# def mol_():
#     mol_h2o = gto.M(unit="Bohr",
#                     atom="O 0.000000000000  -0.143225816552   0.000000000000;"
#                     + "H 1.638036840407   1.136548822547  -0.000000000000;"
#                     + "H -1.638036840407   1.136548822547  -0.000000000000",
#                     basis='STO-3g')
#     return mol_h2o.build()
# def


def test_calc_nuclear_repulsion_energy(mol_h2o):
    assert SCF.calc_nuclear_repulsion_energy(mol_h2o) == 8.00236706181077,\
        "Nuclear Repulsion Energy Test (H2O) Failed"


def test_calc_initial_density(mol_h2o):
    result = SCF.calc_initial_density(mol_h2o)
    assert result.sum() == 0
    assert result.shape == (mol_h2o.nao, mol_h2o.nao),\
        "Calculate innitial density Test (H2O) failed"


def test_calc_hcore_matrix():

    H_core = SCF.calc_hcore_matrix(Tuv, Vuv)
    assert H_core[0, 0] == -32.57739541261037
    assert H_core[3, 4] == 0
    'Calculate hcore matrix Test failed'


def test_calc_fock_matrix():
    results = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)

    assert results[2, 5] == -1.6751501447185013
    assert results[0, 0] == -32.57739541261037

# @pytest.fixture()
# def Fuv():
#     Fuv = SCF.calc_fock_matrix(mol_h2o,Huv,eri,Duv)
#     return Fuv


Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)


def test_solve_Roothan_equations():

    mol_e, mol_c = SCF.solve_Roothan_equations(Fuv, Suv)

    assert mol_e[0] == approx(-32.57830291837955)


mol_e, mol_c = SCF.solve_Roothan_equations(Fuv, Suv)


def test_form_density_matrix():
    Duv_new = SCF.form_density_matrix(mol_h2o, mol_c)
    assert Duv_new[0, 0] == 2.130023428655503


Enuc = SCF.calc_nuclear_repulsion_energy(mol_h2o)
Duv = SCF.calc_initial_density(mol_h2o)


def test_calc_total_energy():
    Etot = SCF.calc_total_energy(Fuv, Huv, Duv, Enuc)
    assert Etot == approx(8.0023670618)
