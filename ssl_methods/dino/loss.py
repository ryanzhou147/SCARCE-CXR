"""
DINOLoss: cross-entropy between teacher and student outputs with centering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """Cross-entropy loss with teacher centering.

    For each (teacher_crop, student_crop) pair excluding the same-view
    pair compute ``-sum(teacher_probs * log(student_probs))`` and average.

    Args:
        out_dim:          Prototype dimension (must match the DINO head).
        n_global_crops:   Number of global crops produced by the teacher.
        student_temp:     Student softmax temperature (higher is softer).
        teacher_temp:     Teacher softmax temperature (lower is sharper).
        center_momentum:  EMA coefficient for the centering buffer.
    """

    def __init__(
        self,
        out_dim: int,
        n_global_crops: int,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.n_global_crops = n_global_crops
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(
        self,
        student_output: list[torch.Tensor],
        teacher_output: list[torch.Tensor],
    ) -> torch.Tensor:
        # Teacher: centre -> temperature -> softmax -> detach (no gradients).
        teacher_probs = [
            F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
            for t in teacher_output
        ]
        # Student: temperature -> log-softmax.
        student_log_probs = [
            F.log_softmax(s / self.student_temp, dim=-1)
            for s in student_output
        ]

        total_loss = torch.tensor(0.0, device=student_output[0].device)
        n_pairs = 0
        for i, t_prob in enumerate(teacher_probs):
            for j, s_lp in enumerate(student_log_probs):
                if i == j:
                    continue  # skip same-view pair
                total_loss -= torch.mean(torch.sum(t_prob * s_lp, dim=-1))
                n_pairs += 1

        self._update_center(teacher_output)
        return total_loss / n_pairs

    @torch.no_grad()
    def _update_center(self, teacher_output: list[torch.Tensor]) -> None:
        """EMA update of the centering buffer from the current teacher outputs."""
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
        self.center = (
            self.center * self.center_momentum
            + batch_center * (1.0 - self.center_momentum)
        )
