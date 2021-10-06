"""
SCF.py is a module that contains all of the functions
for the HF SCF Procedure
"""

import numpy as np
import scipy as sp
import scipy.linalg as linalg

def calc_nuclear_repulsion_energy(mol_):
    """
    calc_nuclear_repulsion_energy - calculates the n-e repulsion energy of a
                                    molecule

    Arguments:
        mol_: the PySCF molecule data structure created from Input

    Returns:
        Enuc: The n-e repulsion energy
    """

    charges = mol_.atom_charges()
    coords = mol_.atom_coords()
    Enuc = 0
    distance_matrix = np.zeros((3, 3), dtype=np.double)

    """
    Replace with your implementation

    Step 1. calcuate (3x3) distance matrix between all atoms
    Step 2. Loop over atoms and calculate Enuc from formulat in Readme
    """

    # step1: distance between atoms
    rows, cols = coords.shape
    distance = []
    for i in range(rows):
        atom_i = coords[i, :]
        distance_row = []
        for j in range(rows):
            atom_j = coords[j, :]
            distance_ij = linalg.norm(atom_i - atom_j)
            distance_row.append(distance_ij)
        distance.append(distance_row)
    distance = np.asarray(distance).reshape(3, 3)

    # step2: loop over atoms for nucleus repusion energy
    for i in range(rows):
        for j in range(i+1, rows, 1):
            distance_ij = distance[i, j]
            charge_i = charges[i]
            charge_j = charges[j]
            e_ij = charge_i * charge_j / distance_ij
            Enuc += e_ij
    return Enuc


def calc_initial_density(mol_):
    """
    calc_initial_density - Function to calculate the initial guess density

    Arguments
        mol_: the PySCF molecule data structure created from Input

    Returns:
        Duv: the (mol.nao x mol.nao) Guess Density Matrix
    """

    num_aos = mol_.nao  # Number of atomic orbitals, dimensions of the mats
    Duv = np.zeros((num_aos, num_aos), dtype=np.double)

    return Duv


def calc_hcore_matrix(Tuv_, Vuv_):
    """
    calc_hcore_matrix - Computes the 1 electron core matrix

    Arguments:
        Tuv_: The Kinetic Energy 1e integral matrix
        Vuv_: The Nuclear Repulsion 1e integrals matrix

    Returns:
        h_core: The one electron hamiltonian matrix
    """

    """
    Replace with your implementation

    Per the readme, this is a simple addition of the two matrices
    """
    h_core = Tuv_ + Vuv_
    return h_core


def calc_fock_matrix(mol_, h_core_, er_ints_, Duv_):
    """
    calc_fock_matrix - Calculates the Fock Matrix of the molecule

    Arguments:
        mol_: the PySCF molecule data structure created from Input
        h_core_: the one electron hamiltonian matrix
        er_ints_: the 2e electron repulsion integrals
        Duv_: the density matrix

    Returns:
        Fuv: The fock matrix

    """

    Fuv = h_core_.copy()  # Takes care of the Huv part of the fock matrix
    num_aos = mol_.nao    # Number of atomic orbitals, dimension of the mats

    """
    Replace with your implementation

    Here you will do the summation of the last two terms in the Fock matrix
    equation involving the two electron Integrals

    Hint: You can do this with explicit loops over matrix indices, whichwill
          have many for loops.

    This can also be done with numpy aggregations, bonus points if you
    implement this with only two loops.

    For example, the first term can be implemented like the following:
    (er_ints[mu,nu]*Duv).sum()
    """
    # repulsion term
    for u in range(num_aos):
        for v in range(num_aos):
            Fuv[u, v] += (Duv_ * er_ints_[u, v]).sum()

    # exchange term
    for u in range(num_aos):
        for v in range(num_aos):
            Fuv[u, v] -= (0.5 * (Duv_ * er_ints_[u, :, v]).sum())

    return Fuv


def solve_Roothan_equations(Fuv_, Suv_):
    """
    solve_Roothan_equations - Solves the matrix equations to determine
                              the MO coefficients

    Arguments:
        Fuv_: The Fock matrix
        Suv_: The overlap matrix

    Returns:
        mo_energies: an array that contains eigenvalues of the solution
        mo_coefficients: a matrix of the eigenvectors of the solution

    """

    """
    Replace with your implementation

    The Roothan Equations, which are of the form FC=SCe can be solved
    directly from the proper use of scipy.linalg.eigh since this is a
    symmetric hermitian matrix. Take a look at the documentation for that
    function and you can implement this in one line.
    """
    mo_energies, mo_coeffs = linalg.eigh(Fuv_, Suv_, eigvals_only=False)
    return mo_energies.real, mo_coeffs.real


def form_density_matrix(mol_, mo_coeffs_):
    """
    form_dentsity_matrix - forms the density matrix from the eigenvectors

    Note: the loops are over the number of electrons / 2, not all of the
    atomic orbitals

    Arguments:
        mol_: the PySCF molecule data structure created from Input
        mo_coefficients: a matrix of the eigenvectors of the solution

    Returns:
        Duv: the density matrix
    """

    nelec = mol_.nelec[0]  # Number of occupied orbitals
    num_aos = mol_.nao  # Number of atomic orbitals, dimensions of the mats
    Duv = np.zeros((mol_.nao, mol_.nao), dtype=np.double)

    """
    Replace with your implementation

    This involves basically a computation of each density matrix element
    that is a sum over the produces of the mo_coeffs.

    """
    m, n = Duv.shape
    for u in range(m):
        for v in range(n):
            for i in range(nelec):
                C_ui = mo_coeffs_[u, i]
                C_vi = mo_coeffs_[v, i]
                Duv[u, v] += (2 * C_ui * C_vi)
    return Duv


def calc_tot_energy(Fuv_, Huv_, Duv_, Enuc_):
    """
    calc_total_energy - This function calculates the total energy of the
    molecular system

    Arguments:
        Fuv_: the current Fock Matrix
        Huv_: the core Hamiltonian Matrix
        Duv_: the Density Matrix that corresponds to Fuv_
        Enuc: the Nuclear Repulsion Energy

    Returns:
        Etot: the total energy of the molecule
    """

    """
    Replace with your implementation

    Should be able to implement this in one line with matrix arithmatic

    """
    Etot = (0.5 * np.sum((Huv_ + Fuv_) * Duv_)) + Enuc_
    return Etot
