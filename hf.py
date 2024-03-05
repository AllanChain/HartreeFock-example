import json
import warnings
from pathlib import Path

import numpy as np
from scipy.special import erf


def erf_with_const(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignore div by zero warning
        return np.where(x == 0, 1, np.sqrt(np.pi / x) * erf(np.sqrt(x)) / 2)


def gnorm(exp):
    """Gaussian normailzation factor."""
    # NOTE: It's important not to forget this factor.
    return (2 * exp / np.pi) ** (3 / 4)


def get_new_center(exp1, center1, exp2, center2):
    # of shape (nbasis, nbasis, ngaussian, ngaussian, ndim)
    return (exp1[..., None] * center1 + exp2[..., None] * center2) / (
        exp1[..., None] + exp2[..., None]
    )


def eval_S_and_H(
    basis: np.ndarray,
    centers: np.ndarray,
    atoms: np.ndarray,
    charges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate essential constants for basis

    Args:
        basis: shape (2, nbasis, ngaussian)
        centers: shape (nbasis, ndim)
        atoms: shape (natom, ndim)
        charges: shape (natom,)

    Returns:
        - overlap matrix S = (A|B)
        - one-body hamiltonian matrix H_core = (A|H|B)
    """

    # after operations will of shape (nbasis, nbasis, ngaussian, ngaussian)
    exp1, coeff1 = basis[:, None, :, None, :]
    exp2, coeff2 = basis[:, :, None, :, None]

    # of shape (nbasis, nbasis, 1, 1, ndim)
    center1 = centers[None, :, None, None, :]
    center2 = centers[:, None, None, None, :]
    # of shape (nbasis, nbasis, 1, 1)
    delta_center = np.linalg.norm(center1 - center2, axis=-1)
    # of shape (nbasis, nbasis, ngaussian, ngaussian, ndim)
    new_center = get_new_center(exp1, center1, exp2, center2)

    harmonic_mean = exp1 * exp2 / (exp1 + exp2)
    result_exp = np.exp(-harmonic_mean * delta_center**2)

    result_coeff = coeff1 * coeff2 * gnorm(exp1) * gnorm(exp2)
    overlap_coeff = result_coeff * (np.pi / (exp1 + exp2)) ** 1.5
    # sum over all gaussian functions in basis function
    overlap_mat = np.sum(overlap_coeff * result_exp, axis=(-2, -1))

    kinetic_coeff = harmonic_mean * (3 - 2 * harmonic_mean * delta_center**2)
    kinetic_mat = np.sum(kinetic_coeff * overlap_coeff * result_exp, axis=(-2, -1))
    print("Kinetic matrix T:", kinetic_mat, sep="\n")

    # of shape (natom, nbasis, nbasis, ngaussian, ngaussian)
    delta_new_center = np.linalg.norm(
        new_center - atoms[:, None, None, None, None, :], axis=-1
    )
    atom_sum = np.sum(
        charges[:, None, None, None, None]
        * erf_with_const((exp1 + exp2) * delta_new_center**2),
        axis=0,
    )
    potential_coeff = result_coeff * -2 * np.pi / (exp1 + exp2)
    potential_mat = np.sum(potential_coeff * atom_sum * result_exp, axis=(-2, -1))
    print("Potential matrix V:", potential_mat, sep="\n")

    return overlap_mat, kinetic_mat + potential_mat


def eval_two_elec_int(basis: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Evaluate two-electron integral (AB|CD)."""

    exp1, coeff1 = basis[:, :, None, None, None, :, None, None, None]
    exp2, coeff2 = basis[:, None, :, None, None, None, :, None, None]
    exp3, coeff3 = basis[:, None, None, :, None, None, None, :, None]
    exp4, coeff4 = basis[:, None, None, None, :, None, None, None, :]

    center1 = centers[:, None, None, None, None, None, None, None, :]
    center2 = centers[None, :, None, None, None, None, None, None, :]
    center3 = centers[None, None, :, None, None, None, None, None, :]
    center4 = centers[None, None, None, :, None, None, None, None, :]

    delta12 = np.linalg.norm(center1 - center2, axis=-1)
    delta34 = np.linalg.norm(center3 - center4, axis=-1)
    new_center_1 = get_new_center(exp1, center1, exp2, center2)
    new_center_2 = get_new_center(exp3, center3, exp4, center4)
    delta_new = np.linalg.norm(new_center_1 - new_center_2, axis=-1)

    result_coeff = coeff1 * coeff2 * coeff3 * coeff4 * 2 * np.pi**2.5
    result_coeff *= gnorm(exp1) * gnorm(exp2) * gnorm(exp3) * gnorm(exp4)
    result_coeff /= (exp1 + exp2) * (exp3 + exp4) * np.sqrt(exp1 + exp2 + exp3 + exp4)
    result_exp = np.exp(
        -exp1 * exp2 / (exp1 + exp2) * delta12**2
        - exp3 * exp4 / (exp3 + exp4) * delta34**2
    )
    result_f = erf_with_const(
        (exp2 + exp2) * (exp3 + exp4) / (exp1 + exp2 + exp3 + exp4) * delta_new**2
    )

    return np.sum(result_coeff * result_exp * result_f, axis=(-4, -3, -2, -1))


def eval_trans_mat(overlap_mat: np.ndarray) -> np.ndarray:
    # eigenvectors[:, i] is the ith eigenvector
    eigenvalues, eigenvectors = np.linalg.eig(overlap_mat)
    return eigenvectors / np.sqrt(eigenvalues)


def eval_fock_mat(
    hamiltonian_mat: np.ndarray, density_mat: np.ndarray, two_elec_int: np.ndarray
) -> np.ndarray:
    j_term = np.einsum("ls,mnsl->mn", density_mat, two_elec_int)
    k_term = -np.einsum("ls,mlsn->mn", density_mat, two_elec_int) / 2
    return hamiltonian_mat + j_term + k_term


def eval_density_mat(coeff_mat: np.ndarray, occupied_orbitals: int) -> np.ndarray:
    coeff_occupied = coeff_mat[:, :occupied_orbitals]
    return 2 * np.sum(coeff_occupied.conj() * coeff_occupied[:, None, :], axis=-1)


def read_HeH_basis():
    with open(Path(__file__).parent / "sto-3g.json") as f:
        sto3g = json.load(f)["elements"]

    h_exp = sto3g["1"]["electron_shells"][0]["exponents"]
    h_coeff = sto3g["1"]["electron_shells"][0]["coefficients"][0]
    he_exp = sto3g["2"]["electron_shells"][0]["exponents"]
    he_coeff = sto3g["2"]["electron_shells"][0]["coefficients"][0]

    return np.array(
        [
            [made_float(h_exp), made_float(he_exp)],
            [made_float(h_coeff), made_float(he_coeff)],
        ]
    )


def made_float(data):
    return [float(x) for x in data]


def solve_HeH():
    basis = read_HeH_basis()
    nbasis = basis.shape[1]
    bond_length = 1.4632
    centers = atoms = np.array([[0.0, 0.0, 0.0], [bond_length, 0.0, 0.0]])
    charges = np.array([1, 2])
    nuc_energy = 2 / bond_length
    density_mat = np.zeros((nbasis, nbasis))
    overlap_mat, hamiltonian_mat = eval_S_and_H(basis, centers, atoms, charges)
    two_elec_int = eval_two_elec_int(basis, centers)
    transform_mat = eval_trans_mat(overlap_mat)
    print("Overlap matrix S:", overlap_mat, sep="\n")
    print("Hamiltonian matrix H_core:", hamiltonian_mat, sep="\n")
    print("Transform matrix X:", transform_mat, sep="\n")
    while True:
        fock_mat = eval_fock_mat(hamiltonian_mat, density_mat, two_elec_int)
        transformed_fock = transform_mat.T.conj() @ fock_mat @ transform_mat
        orbital_energies, transformed_mos = np.linalg.eig(transformed_fock)
        # NOTE: If HeH is working but HHe is not, try sorting the eigenvectors.
        # Because the occupied orbitals sort come first to make density matrix correct.
        energy_sort = np.argsort(orbital_energies)
        orbital_energies = orbital_energies[energy_sort]
        mos = transform_mat @ transformed_mos[:, energy_sort]
        new_density_mat = eval_density_mat(mos, occupied_orbitals=1)
        total_energy = np.trace(density_mat @ (hamiltonian_mat + fock_mat)) / 2
        # FIXME: They are old steps
        print("Total energy:", total_energy + nuc_energy)
        print("Orbital energies:", orbital_energies)
        print("Occupation:", np.diag(density_mat * overlap_mat))
        if np.allclose(new_density_mat, density_mat, atol=1e-6):
            break
        density_mat = new_density_mat


if __name__ == "__main__":
    solve_HeH()
