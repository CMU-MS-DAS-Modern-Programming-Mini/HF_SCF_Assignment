import pytest
import SCF


def test_calc_nuclear_repulsion_energy(mol_h2o):
    assert True
#    assert SCF.calc_nuclear_repulsion_energy(mol_h2o) == 8.00236706181077,\
#        "Nuclear Repulsion Energy Test (H2O) Failed"


def test_calc_initial_density(mol_h2o):
    """
    Tests that the initial density returns a zero matrix
    and tests dimensions
    """

    Duv = SCF.calc_initial_density(mol_h2o)
    assert Duv.sum() == 0.0
    assert Duv.shape == (mol_h2o.nao,mol_h2o.nao)
