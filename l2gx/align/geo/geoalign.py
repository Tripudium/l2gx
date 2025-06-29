"""
Alignment using pytorch geometric.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim

from l2gx.align.registry import register_aligner
from l2gx.align.alignment import AlignmentProblem
from l2gx.patch import Patch
from l2gx.align.geo.model import AffineModel, OrthogonalModel
from l2gx.align.utils import to_device

def patchgraph_mse_loss(transformed_emb):
    """
    Custom loss function that computes the squared norm of differences
    between transformed pairs in the dictionary.

    Args:
        transformed_dict: Dictionary with keys (i,j) and values (XW_i+b_i, YW_j+b_j)

    Returns:
        Total loss as the sum of squared differences
    """
    total_loss = 0.0

    for (_, _), (transformed_X, transformed_Y) in transformed_emb.items():
        # Calculate squared norm of the difference
        pair_loss = F.mse_loss(transformed_X, transformed_Y, reduction="mean")
        total_loss += pair_loss

    return total_loss

@register_aligner("geo")
class GeoAlignmentProblem(AlignmentProblem):
    """
    Alignment problem using pytorch geometric.
    """

    def __init__(
        self,
        num_epochs: int = 1000,
        learning_rate: float = 0.001,
        model_type: str = "affine",
        device: str = "cpu",
        verbose=False
    ):
        super().__init__(
            verbose
        )
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.device = device
        self.model_type = model_type
        self.loss_hist = []

    def set_learning_rate(self, learning_rate: float):
        self.learning_rate = learning_rate

    def set_num_epochs(self, num_epochs: int):
        self.num_epochs = num_epochs

    def _get_intersections(self):
        """
        Calculate the intersection of nodes between patches.
        """
        intersections = {}
        embeddings = {}
        for i, _ in enumerate(self.patches):
            for j in range(len(self.patches)):
                if i < j:
                    intersections[(i, j)] = list(
                        set(self.patches[i].nodes.tolist()).intersection(
                            set(self.patches[j].nodes.tolist())
                        )
                    )
                    if len(intersections[(i, j)]) >= self.min_overlap:
                        embeddings[(i, j)] = [
                            torch.tensor(
                                self.patches[i].get_coordinates(list(intersections[(i, j)]))
                            ),
                            torch.tensor(
                                self.patches[j].get_coordinates(list(intersections[(i, j)]))
                            ),
                        ]
        # embeddings = list(itertools.chain.from_iterable(embeddings))
        return intersections, embeddings
    
    def train_alignment_model(
        self,
        embeddings,
        device="cpu",
        num_epochs=100,
        learning_rate=0.05,
        model_type="affine",
        verbose=True,
    ):
        """
        Train the model on the patch embeddings
        Args:
            patch_emb: list of torch.Tensor
                patch embeddings
            n_patches: int
                number of patches
            device: str
                device to run the model on
            num_epochs: int
                number of epochs to train the model
            learning_rate: float
                learning rate for the optimizer
        Returns:
            model: Model
            loss_hist: list
        """
        patch_emb = to_device(embeddings, device)
        dim = patch_emb[list(patch_emb.keys())[0]][0].shape[1]
        model = (
            AffineModel(dim, self.n_patches, device).to(device)
            if model_type == "affine"
            else OrthogonalModel(dim, self.n_patches, device).to(device)
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_hist = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            transformed_patch_emb = model(patch_emb)
            loss = patchgraph_mse_loss(transformed_patch_emb)
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_hist.append(loss.item())
            if verbose:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item()}")

        return model, loss_hist

    def align_patches(self, patches: list[Patch], min_overlap: int | None = None):
        """
        Align the patches.
        """
        self._register_patches(patches, min_overlap)
        intersections, embeddings = self._get_intersections()
        self.intersections = intersections

        res, loss_hist = self.train_alignment_model(
            embeddings,
            device=self.device,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            model_type=self.model_type,
            verbose=self.verbose,
        )

        self.loss_hist = loss_hist

        self.rotations = [
            res.transformation[i].weight.to("cpu").detach().numpy()
            for i in range(self.n_patches)
        ]
        self.shifts = [
            res.transformation[i].bias.to("cpu").detach().numpy()
            for i in range(self.n_patches)
        ]
        for i, patch in enumerate(self.patches):
            self.patches[i].coordinates = patch.coordinates @ self.rotations[i].T
            self.patches[i].coordinates += self.shifts[i]
        self._aligned_embedding = self.mean_embedding()