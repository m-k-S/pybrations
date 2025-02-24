import torch
from fairchem.core import OCPCalculator
from fairchem.core.common.registry import registry
from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.datasets import data_list_collater
from fairchem.core.trainers.ocp_trainer import OCPTrainer

from fairchem.core.models.base import HydraModel
from fairchem.core.models.equiformer_v2.equiformer_v2 import EquiformerV2Backbone
from fairchem.core.models.equiformer_v2.prediction_heads.rank2 import Rank2SymmetricTensorHead

from collections import defaultdict
from tqdm import tqdm

class MinimalOCPCalculator:
    def __init__(self, checkpoint_path, cpu=True):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        config = checkpoint["config"]

        # Set up trainer
        self.trainer = OCPTrainer(
            task=config.get("task", {}),
            model=config["model"],
            dataset=[config["dataset"]],
            outputs=config["outputs"],
            loss_functions=config["loss_functions"],
            evaluation_metrics=config["evaluation_metrics"],
            optimizer=config["optim"],
            identifier="",
            slurm=config.get("slurm", {}),
            local_rank=config.get("local_rank", 0),
            is_debug=config.get("is_debug", True),
            cpu=cpu,
            amp=config.get("amp", False),
            inference_only=True,
        )
        
        # Load model weights
        self.trainer.load_checkpoint(checkpoint_path)
        
        # Initialize graph converter
        self.converter = AtomsToGraphs(
            max_neigh=50,
            radius=6,
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_edges=False,
            r_pbc=True
        )
    
    def predict(self, atoms):
        """
        Args:
            atoms
        Returns:
            dict with 'energy' and 'forces' keys
        """  
        # Convert to graph
        data_object = self.converter.convert(atoms)
        batch_object = data_list_collater([data_object], otf_graph=True)

        self.trainer.model.eval()
        
        batch_object.pos.requires_grad = True

        with torch.amp.autocast('cuda', enabled=self.trainer.scaler is not None):
            out = self.trainer._forward(batch_object)

        # target_key = 'forces'
        # pred = self.trainer._denorm_preds(target_key, out[target_key], batch)

        # batch_natoms, batch_fixed = batch.natoms, batch.fixed
        # per_image_pred = torch.split(pred, batch_natoms.tolist())
        # per_image_free_preds = [_pred[(fixed == 0).tolist()] for _pred, fixed in zip(per_image_pred, torch.split(batch_fixed, batch_natoms.tolist()))]
        # _chunk_idx = [len(free_pred) for free_pred in per_image_free_preds]
        # predictions[target_key].extend(per_image_free_preds)
        # predictions["chunk_idx"].extend(_chunk_idx)

        return batch_object.pos, out['forces']


# Load atoms from CIF file
from ase.io import read

# Read the AgI structure from CIF file
atoms = read('AgI.cif')

ocp = MinimalOCPCalculator('models/eqV2_31M_omat_mp_salex.pt')

# Make prediction
positions, forces = ocp.predict(atoms)
n_atoms = positions.shape[0]

# may want to reshape this in the future
hessian = torch.zeros((n_atoms * 3, n_atoms * 3), dtype=positions.dtype, device=positions.device)

for i in range(n_atoms):
    for alpha in range(3):  # x, y, z
        force_component = forces[i, alpha]  # Select F_i^α

        # Compute first derivative dF_i^α / d r_j^β
        grad_outputs = torch.zeros_like(forces)
        grad_outputs[i, alpha] = 1.0  # Select specific component
        grad_first = torch.autograd.grad(forces, positions, grad_outputs=grad_outputs, create_graph=True)[0]

        for j in range(n_atoms):
            for beta in range(3):
                hessian[i * 3 + alpha, j * 3 + beta] = grad_first[j, beta]  # Store dF_i^α / dr_j^β

# Save the Hessian matrix to a torch file
torch.save(hessian, 'hessian.pt')
print (hessian.shape)

import ase 
import numpy as np

masses = ase.data.atomic_masses[atoms.get_atomic_numbers()]

npoints = 100

rec_vecs = 2 * np.pi * atoms.cell.reciprocal().real
mp_band_path = atoms.cell.bandpath(npoints=npoints)

all_kpts = mp_band_path.kpts @ rec_vecs
all_eigs = []

def sqrt(x):
    return np.sign(x) * np.sqrt(np.abs(x))

def dynamical_matrix(kpt, atoms, H, masses):
    r"""Dynamical matrix at a given k-point.

    .. math::

        D_{ij}(\vec k) = \hat H_{ij}(\vec k) / \sqrt{m_i m_j}

    """
    r = atoms.get_positions()
    ph = np.exp(-1j * np.dot(r[:, None, :] - r[None, :, :], kpt))[:, None, :, None]
    # a = graph.nodes.index_cell0
    a = [i for i in range(len(r))]
    i = np.arange(3)
    Hk = (
        np.zeros((n_atoms, 3, n_atoms, 3), dtype=ph.dtype)
        .at[np.ix_(a, i, a, i)]
        .add(ph * H)
    )
    Hk = Hk.reshape((masses.size, 3, masses.size, 3))

    iM = 1 / np.sqrt(masses)
    Hk = np.einsum("i,iujv,j->iujv", iM, Hk, iM)
    Hk = Hk.reshape((3 * masses.size, 3 * masses.size))
    return Hk

for kpt in tqdm(all_kpts):
    Dk = dynamical_matrix(kpt, atoms, hessian, masses)
    Dk = np.asarray(Dk)
    all_eigs.append(np.sort(sqrt(np.linalg.eigh(Dk)[0])))

#     all_eigs = np.stack(all_eigs)

#     eV_to_J = 1.60218e-19
#     angstrom_to_m = 1e-10
#     atom_mass = 1.660599e-27  # kg
#     hbar = 1.05457182e-34
#     cm_inv = (0.124e-3) * (1.60218e-19)  # in J
#     conv_const = eV_to_J / (angstrom_to_m**2) / atom_mass

#     all_eigs = all_eigs * np.sqrt(conv_const) * hbar / cm_inv

#     bs = ase.spectrum.band_structure.BandStructure(mp_band_path, all_eigs[None])

#     plt.figure(figsize=(7, 6), dpi=100)
#     bs.plot(ax=plt.gca(), emin=1.1 * np.min(all_eigs), emax=1.1 * np.max(all_eigs))
#     plt.ylabel("Phonon Frequency (cm$^{-1}$)")
#     plt.tight_layout()
#     return plt.gcf()