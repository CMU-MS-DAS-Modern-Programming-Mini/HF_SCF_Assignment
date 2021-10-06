"""
This is the Hartree-Fock SCF Assignmentthat is for the class assignment for
Modern Programming in Data Analytics

It is intended to give students a basic exposure to the following concepts

1. Working with novel libraries
2. Converting math to python programming
3. Basic exposure to using NumPy and SciPy
4. Proper documentation of code and unit testing

The instructions for the assignement come in the Readme from the repository
at:
"""
import SCF
import numpy as np
import pickle
import sys


class mol:
    """
    class mol

    Contains the information about the molecule
    """

    def __init__(self, atom_string_, nao_, nelect_, nmo_):
        """
        mol Creator - creates an instance of the mol class

        Arguments:
          atoms_string_: A string for defining the atoms in the molecule
          nao_: Number of atomic orbitals
          nelec_: Number of electrons / 2
          nmo_: Number of molecular orbitals

        Returns:
            instance of mol
        """
        self.nao = nao_
        self.nelec = [nelect_, nelect_]
        self.nmo = nmo_
        self.charges = []
        coords_list = []
        atom_list = atom_string_.split(";")
        for a in atom_list:
            print(a.split()[2])
            self.charges.append(int(a.split()[0]))
            coords_list.append([float(a.split()[1]),
                                float(a.split()[2]),
                                float(a.split()[3])])
        self.coords = np.array(coords_list)

    def atom_charges(self):
        return self.charges

    def atom_coords(self):
        return self.coords


def main():
    """
    The main function call, meant to gather input and invoke the
    Hartree Fock Self Consistent Field Procedure

    Output:
        The total Hartee-Fock Energy of the molecule
    """

    # Input Data
    atom = "8 0.000000000000  -0.143225816552   0.000000000000;" \
        + "1 1.638036840407   1.136548822547  -0.000000000000;" \
        + "1 -1.638036840407   1.136548822547  -0.000000000000"

    mol_h2o = mol(atom, 7, 5, 7)
    # Convergence Criteria
    E_conv_threshold = 1.0E-10
    D_conv_threshold = 1.0E-8
    max_iterations = 1000

    # get the integrals
    # Overlap Integrals
    Suv = pickle.load(open("suv.pkl", "rb"))
    # Kinetic Energy 1 electron integrals
    Tuv = pickle.load(open("tuv.pkl", "rb"))
    # Nuclear Repulsion 1 electron integrals
    Vuv = pickle.load(open("vuv.pkl", "rb"))
    # Electron Repulsion 2 electron integrals
    eri = pickle.load(open("eri.pkl", "rb"))

    """
    Main SCF Procedure

    Each step is outlined in the Github Repo Readme
    Your job will be to fill in the stubbed functions in SCF.py

    Step 1. Calculate the Electron Nuclear Repulsion Energy
    Step 2. Calculate the Initial Hcore Matrix
    Step 3. Calculate the Inital Density Matrix
    Step 4. Start the SCF Procedure
        Step 4a. Calculate the Fock Matrix
        Step 4b. Solve Eigenvalues and Eigenvectors of Roothan Equations
        Step 4c. Calculate the Total Energy of the Current Iteration
        Step 4d. Caluate the new Density Matrix
        Step 4e. Calulate the Energy Difference and RMS Difference of Density
        Step 4f. Check for Convergence, if Converged, Exit
        Step 4g. If not Converged, update Density Matrix and Energy and do
                 another iteration
    Step 5. Print out Final Total Energy for User
    """

    Enuc = SCF.calc_nuclear_repulsion_energy(mol_h2o)
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    Duv = SCF.calc_initial_density(mol_h2o)
    Etot = 0.0
    for it in range(max_iterations):
        Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)

        mo_e, mo_c = SCF.solve_Roothan_equations(Fuv, Suv)

        Etot_new = SCF.calc_total_energy(Fuv, Huv, Duv, Enuc)

        Duv_new = SCF.form_density_matrix(mol_h2o, mo_c)

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


if __name__ == "__main__":
    main()
