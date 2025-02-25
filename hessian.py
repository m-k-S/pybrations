
import ase 
import numpy as np
from matplotlib import pyplot as plt
import torch

from tqdm import tqdm

import ase
import ase.spectrum.band_structure

from ase.io import read

from create_graphs import extended_graph


def dynamical_matrix(kpt, r, a, H, masses):
    
    print (H.shape)

    # Compute phase factor
    displacement = r[:, None, :] - r[None, :, :]  # Shape: (n_atoms, n_atoms, 3)
    phase = torch.exp(-1j * torch.einsum('ijk,k->ij', displacement, kpt))  # Shape: (n_atoms, n_atoms)
    phase = phase[:, None, :, None]  # Reshape to match original NumPy behavior

    print (phase.shape)

    n_atoms = len(r)
    Hk = torch.zeros((n_atoms, 3, n_atoms, 3), dtype=phase.dtype, device=phase.device)

    i = torch.arange(3)

    Hk.index_add_(0, a, phase * H)  # Accumulate values
    Hk = Hk.reshape((masses.numel(), 3, masses.numel(), 3))

    iM = 1 / torch.sqrt(masses)
    Hk = torch.einsum("i,iujv,j->iujv", iM, Hk, iM)
    Hk = Hk.reshape((3 * masses.numel(), 3 * masses.numel()))

    return Hk


# Read the AgI structure from CIF file
atoms = read('AgI.cif')

H = torch.load("hessian.pt")

masses = ase.data.atomic_masses[atoms.get_atomic_numbers()]

npoints = 100

rec_vecs = 2 * np.pi * atoms.cell.reciprocal().real
mp_band_path = atoms.cell.bandpath(npoints=npoints)

# print (get_neighborhood(atoms.get_positions(), 5.0, atoms.pbc, atoms.cell))

extended_positions, extended_indices = extended_graph(atoms)
print (extended_positions.shape)
extended_positions = torch.tensor(extended_positions, dtype=torch.complex128)
extended_indices = torch.tensor(extended_indices, dtype=torch.complex128)

all_kpts = mp_band_path.kpts @ rec_vecs
# all_eigs = []

dk = dynamical_matrix(all_kpts[0], extended_positions, extended_indices, H, masses)

print (dk.shape)

# for kpt in tqdm(all_kpts):
#     Dk = dynamical_matrix(kpt, atoms, H, masses)
#     # Dk = np.asarray(Dk)
#     Dk = Dk.detach().numpy()
#     all_eigs.append(np.sort(sqrt(np.linalg.eigh(Dk)[0])))

# all_eigs = np.stack(all_eigs)

# eV_to_J = 1.60218e-19
# angstrom_to_m = 1e-10
# atom_mass = 1.660599e-27  # kg
# hbar = 1.05457182e-34
# cm_inv = (0.124e-3) * (1.60218e-19)  # in J
# conv_const = eV_to_J / (angstrom_to_m**2) / atom_mass

# all_eigs = all_eigs * np.sqrt(conv_const) * hbar / cm_inv

# bs = ase.spectrum.band_structure.BandStructure(mp_band_path, all_eigs[None])

# plt.figure(figsize=(7, 6), dpi=100)
# bs.plot(ax=plt.gca(), emin=1.1 * np.min(all_eigs), emax=1.1 * np.max(all_eigs))
# plt.ylabel("Phonon Frequency (cm$^{-1}$)")
# plt.tight_layout()
# plt.show()