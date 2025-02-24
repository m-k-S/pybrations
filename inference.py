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

        grad_first = torch.autograd.grad(out['forces'].sum(), batch_object.pos, allow_unused=True)[0]
        print(grad_first)  # Should not be None 

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
# results = ocp.predict(atoms)
positions, forces = ocp.predict(atoms)
# forces = results['forces'][0] # if there is only one material
# positions = results['positions'][0] # if there is only one material

grad_first = torch.autograd.grad(forces.sum(), positions, allow_unused=True)[0]
print(grad_first)  # Should not be None


# n_atoms = positions.shape[0]
# hessian = torch.zeros((n_atoms * 3, n_atoms * 3), dtype=positions.dtype, device=positions.device)

# for i in range(n_atoms):
#     for alpha in range(3):  # x, y, z
#         force_component = forces[i, alpha]  # Select F_i^α

#         # Compute first derivative dF_i^α / d r_j^β
#         grad_outputs = torch.zeros_like(forces)
#         grad_outputs[i, alpha] = 1.0  # Select specific component
#         grad_first = torch.autograd.grad(forces, positions, grad_outputs=grad_outputs, create_graph=True)[0]

#         for j in range(n_atoms):
#             for beta in range(3):
#                 hessian[i * 3 + alpha, j * 3 + beta] = grad_first[j, beta]  # Store dF_i^α / dr_j^β

# print (hessian)