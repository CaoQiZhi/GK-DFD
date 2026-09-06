from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEmbed(nn.Module):
    """Map a branch feature to the graph node embedding space."""

    def __init__(self, dim_in, dim_out):
        super(FeatureEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, feature):
        feature = feature.view(feature.size(0), -1)
        return F.normalize(self.linear(feature), p=2, dim=1)


class DenseTAGConv(nn.Module):
    """A dense one-hop TAG-style graph convolution."""

    def __init__(self, dim):
        super(DenseTAGConv, self).__init__()
        self.weight = nn.Parameter(torch.empty(2, dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in = self.weight.size(1)
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, feature, adjacency):
        degree = adjacency.sum(dim=1).clamp_min(1e-6)
        degree = degree.rsqrt()
        normalized = degree.unsqueeze(1) * adjacency * degree.unsqueeze(0)
        propagated = torch.mm(normalized, feature)
        output = torch.mm(feature, self.weight[0])
        output = output + torch.mm(propagated, self.weight[1]) + self.bias
        return F.normalize(output, p=2, dim=1)


class GKDFDLoss(nn.Module):
    """Graph Knowledge guided Discriminative Feature Distillation.

    The implementation follows Sections 3.2--3.4 of the paper: teacher
    probabilities and labels build one shared UEM/CAC topology; centered
    teacher PCA defines a shared coordinate system; signed local alignment
    matrices define the DFD eigenspace.
    """

    def __init__(self, opt):
        super(GKDFDLoss, self).__init__()
        self.embed_s = FeatureEmbed(opt.s_dim, opt.feat_dim)
        self.embed_t = FeatureEmbed(opt.t_dim, opt.feat_dim)
        self.gnn_s = DenseTAGConv(opt.feat_dim)
        self.gnn_t = DenseTAGConv(opt.feat_dim)

        self.cac_scale = float(opt.gkdfd_cac_scale)
        self.pca_ratio = float(opt.gkdfd_pca_ratio)
        self.positive_neighbors = int(opt.gkdfd_k1)
        k2 = getattr(opt, 'gkdfd_k2', None)
        self.negative_neighbors = None if k2 is None else int(k2)
        self.dla_dim = int(opt.gkdfd_dla_dim)

    @staticmethod
    def _eigh(matrix):
        matrix = 0.5 * (matrix + matrix.t())
        if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'eigh'):
            return torch.linalg.eigh(matrix)
        return torch.symeig(matrix, eigenvectors=True)

    def _uem_adjacency(self, logits, target=None):
        """Build the shared adjacency using equations (3)--(7)."""
        # Section 3.2.1 defines p_i as the model logits.  UEM therefore uses
        # the raw teacher logits, rather than the softmax probabilities used
        # by the older HKD prototype.
        prediction = logits.detach()
        normalized = F.normalize(prediction, p=2, dim=1)
        cosine = torch.mm(normalized, normalized.t()).clamp(-1.0, 1.0)
        cosine = (1.0 + cosine) * 0.5
        euclidean = torch.cdist(prediction, prediction, p=2)
        exp_distance = 1.0 - torch.exp(-euclidean)
        edge = cosine - exp_distance

        # ``target`` is required by CAC in the paper.  Falling back to the
        # teacher argmax keeps this helper backward-compatible for callers
        # that used the pre-paper prototype directly.
        if target is None:
            target = prediction.argmax(dim=1)
        labels = target.detach().view(-1)
        same_class = labels.unsqueeze(1).eq(labels.unsqueeze(0))
        edge = torch.where(same_class, edge, self.cac_scale * edge)
        # Equation (7) retains only positive affinities; the positive diagonal
        # is kept as a TAGConv self connection.
        return edge.clamp_min(0.0)

    def _graph_representation(self, feature, adjacency, embed, graph_encoder):
        node = embed(feature)
        graph = graph_encoder(node, adjacency)
        return node, graph

    def _pca_project(self, teacher, student):
        """Center both branches and project with teacher-fitted PCA (eq. 9)."""
        with torch.no_grad():
            center = teacher.detach().mean(dim=0, keepdim=True)
            centered = teacher.detach() - center
            covariance = torch.mm(centered.t(), centered)
            covariance = covariance / float(max(teacher.size(0) - 1, 1))
            _, eigenvectors = self._eigh(covariance)
            max_rank = min(teacher.size(1), max(teacher.size(0) - 1, 1))
            reduced_dim = max(1, int(round(teacher.size(1) / self.pca_ratio)))
            reduced_dim = min(reduced_dim, max_rank)
            projection = eigenvectors[:, -reduced_dim:]

        teacher_low = torch.mm(teacher - center, projection)
        student_low = torch.mm(student - center, projection)
        return teacher_low, student_low

    def _select_neighbors(self, reference, target):
        """Select nearest same-/different-class neighbours for every anchor."""
        count = reference.size(0)
        distance = torch.cdist(reference.detach(), reference.detach(), p=2)
        selected = []
        for index in range(count):
            positive = torch.nonzero(
                target.eq(target[index]), as_tuple=False
            ).view(-1)
            positive = positive[positive.ne(index)]
            negative = torch.nonzero(
                target.ne(target[index]), as_tuple=False
            ).view(-1)

            k1 = min(self.positive_neighbors, positive.numel())
            k2_limit = negative.numel() if self.negative_neighbors is None else self.negative_neighbors
            k2 = min(k2_limit, negative.numel())
            if k1:
                positive = positive[torch.topk(
                    distance[index, positive], k1, largest=False
                ).indices]
            else:
                positive = positive[:0]
            if k2:
                negative = negative[torch.topk(
                    distance[index, negative], k2, largest=False
                ).indices]
            else:
                negative = negative[:0]
            selected.append((positive, negative))
        return selected

    def _relation_weights(self, reference, target):
        """Assemble L=sum_i S_i L_i S_i^T from equations (13)--(18)."""
        count = reference.size(0)
        alignment = reference.new_zeros((count, count))
        for anchor, (positive, negative) in enumerate(
            self._select_neighbors(reference, target)
        ):
            k1 = positive.numel()
            k2 = negative.numel()
            if k1 == 0 and k2 == 0:
                continue
            theta = float(k1) / float(k2) if k2 else 0.0
            weights = reference.new_empty(k1 + k2)
            if k1:
                weights[:k1] = 1.0
            if k2:
                weights[k1:] = -theta

            indices = torch.cat([
                reference.new_tensor([anchor], dtype=torch.long),
                positive,
                negative,
            ])
            local = reference.new_zeros((indices.numel(), indices.numel()))
            local[0, 0] = weights.sum()
            local[0, 1:] = -weights
            local[1:, 0] = -weights
            local[1:, 1:] = torch.diag(weights)
            alignment[indices.unsqueeze(1), indices.unsqueeze(0)] += local

        return 0.5 * (alignment + alignment.t())

    def _discriminative_basis(self, teacher, alignment, target):
        """Return detached eigenvectors for the d smallest eigenvalues."""
        with torch.no_grad():
            objective = torch.mm(teacher.detach().t(), alignment)
            objective = torch.mm(objective, teacher.detach())
            _, eigenvectors = self._eigh(objective)
            if self.dla_dim > 0:
                output_dim = self.dla_dim
            else:
                output_dim = max(target.detach().unique().numel() - 1, 1)
            output_dim = min(output_dim, teacher.size(1))
            return eigenvectors[:, :output_dim].detach()

    def _discriminative_loss(self, teacher, student, target):
        teacher_low, student_low = self._pca_project(teacher, student)
        with torch.no_grad():
            alignment = self._relation_weights(teacher_low.detach(), target)
            basis = self._discriminative_basis(
                teacher_low.detach(), alignment, target
            )
        teacher_projected = torch.mm(teacher_low, basis)
        student_projected = torch.mm(student_low, basis)
        return (teacher_projected - student_projected).pow(2).sum(dim=1).mean()

    def forward(self, feature_s, logit_s, feature_t, logit_t, target):
        # Algorithm 1, step 3: teacher predictions and labels define one
        # topology shared by both graph branches.
        adjacency = self._uem_adjacency(logit_t, target)
        _, graph_s = self._graph_representation(
            feature_s, adjacency, self.embed_s, self.gnn_s
        )
        _, graph_t = self._graph_representation(
            feature_t.detach(), adjacency, self.embed_t, self.gnn_t
        )

        # Equation (22), normalized Frobenius graph consistency loss.
        loss_graph = (graph_t - graph_s).pow(2).sum() / float(
            graph_t.size(0) * graph_t.size(1)
        )
        loss_feature = self._discriminative_loss(graph_t, graph_s, target)
        return loss_graph, loss_feature
