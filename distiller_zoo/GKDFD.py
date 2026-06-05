from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEmbed(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(FeatureEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, feature):
        feature = feature.view(feature.size(0), -1)
        return F.normalize(self.linear(feature), p=2, dim=1)


class DenseTAGConv(nn.Module):
    """One-hop topology adaptive graph convolution without a DGL dependency."""

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
        adjacency = degree.unsqueeze(1) * adjacency * degree.unsqueeze(0)
        propagated = torch.mm(adjacency, feature)
        output = torch.mm(feature, self.weight[0])
        output = output + torch.mm(propagated, self.weight[1]) + self.bias
        return F.normalize(output, p=2, dim=1)


class GKDFDLoss(nn.Module):
    """Graph Knowledge based Discriminative Feature Distillation."""

    def __init__(self, opt):
        super(GKDFDLoss, self).__init__()
        self.embed_s = FeatureEmbed(opt.s_dim, opt.feat_dim)
        self.embed_t = FeatureEmbed(opt.t_dim, opt.feat_dim)
        self.gnn_s = DenseTAGConv(opt.feat_dim)
        self.gnn_t = DenseTAGConv(opt.feat_dim)

        self.cac_scale = opt.gkdfd_cac_scale
        self.pca_ratio = opt.gkdfd_pca_ratio
        self.positive_neighbors = opt.gkdfd_k1
        self.inter_weight = opt.gkdfd_inter_weight
        self.dla_dim = opt.gkdfd_dla_dim

    @staticmethod
    def _eigh(matrix):
        matrix = 0.5 * (matrix + matrix.t())
        if hasattr(torch, 'linalg') and hasattr(torch.linalg, 'eigh'):
            return torch.linalg.eigh(matrix)
        return torch.symeig(matrix, eigenvectors=True)

    def _uem_adjacency(self, logits):
        prediction = F.softmax(logits, dim=1)
        normalized = F.normalize(prediction, p=2, dim=1)
        cosine = torch.mm(normalized, normalized.t()).clamp(-1.0, 1.0)
        cosine = (1.0 + cosine) * 0.5

        euclidean = torch.cdist(prediction, prediction, p=2)
        euclidean = 1.0 - torch.exp(-euclidean)
        edge = cosine - euclidean

        predicted_class = prediction.argmax(dim=1)
        same_class = predicted_class.unsqueeze(1).eq(predicted_class.unsqueeze(0))
        edge = torch.where(same_class, edge, self.cac_scale * edge)
        return edge.clamp_min(0.0)

    def _graph_representation(self, feature, logits, embed, graph_encoder):
        node = embed(feature)
        edge = self._uem_adjacency(logits)
        graph = graph_encoder(node, edge)
        return node, edge, graph

    def _pca_project(self, teacher, student):
        with torch.no_grad():
            center = teacher.detach().mean(dim=0, keepdim=True)
            centered = teacher.detach() - center
            covariance = torch.mm(centered.t(), centered)
            covariance = covariance / float(max(teacher.size(0) - 1, 1))

            eigenvalues, eigenvectors = self._eigh(covariance)
            max_rank = min(
                teacher.size(1),
                max(teacher.size(0) - 1, 1),
            )
            reduced_dim = max(1, int(round(teacher.size(1) / self.pca_ratio)))
            reduced_dim = min(reduced_dim, max_rank)
            eigenvalues = eigenvalues[-reduced_dim:].clamp_min(1e-5)
            eigenvectors = eigenvectors[:, -reduced_dim:]
            projection = eigenvectors * eigenvalues.rsqrt().unsqueeze(0)

        teacher_low = torch.mm(teacher - center, projection)
        student_low = torch.mm(student - center, projection)
        return teacher_low, student_low

    def _relation_weights(self, teacher, target):
        count = teacher.size(0)
        distance = torch.cdist(teacher, teacher, p=2)
        weights = teacher.new_zeros(count, count)
        weights.fill_diagonal_(1.0)

        for index in range(count):
            positive = torch.nonzero(
                target.eq(target[index]), as_tuple=False
            ).view(-1)
            positive = positive[positive.ne(index)]
            negative = torch.nonzero(
                target.ne(target[index]), as_tuple=False
            ).view(-1)

            if positive.numel() > 0:
                k1 = min(self.positive_neighbors, positive.numel())
                order = torch.topk(
                    distance[index, positive], k1, largest=False
                ).indices
                positive = positive[order]
                weights[index, positive] = 1.0
            else:
                k1 = 0

            if negative.numel() > 0:
                if self.inter_weight > 0:
                    theta = self.inter_weight
                else:
                    theta = float(max(k1, 1)) / float(negative.numel())
                weights[index, negative] = -theta

        return weights

    def _discriminative_basis(self, teacher, weights, target):
        with torch.no_grad():
            relation = 0.5 * (weights + weights.t())
            relation.fill_diagonal_(0.0)
            laplacian = torch.diag(relation.sum(dim=1)) - relation
            objective = torch.mm(teacher.detach().t(), laplacian)
            objective = torch.mm(objective, teacher.detach())
            _, eigenvectors = self._eigh(objective)

            if self.dla_dim > 0:
                output_dim = self.dla_dim
            else:
                output_dim = max(target.detach().unique().numel() - 1, 1)
            output_dim = min(output_dim, teacher.size(1))
            return eigenvectors[:, :output_dim]

    def _discriminative_loss(self, teacher, student, target):
        teacher, student = self._pca_project(teacher, student)
        with torch.no_grad():
            weights = self._relation_weights(teacher.detach(), target)
            basis = self._discriminative_basis(
                teacher.detach(), weights, target
            )

        teacher = F.normalize(torch.mm(teacher, basis), p=2, dim=1)
        student = F.normalize(torch.mm(student, basis), p=2, dim=1)
        distance = (
            teacher.pow(2).sum(dim=1, keepdim=True)
            + student.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * torch.mm(teacher, student.t())
        ).clamp_min(0.0)
        return (distance * weights).sum() / weights.abs().sum().clamp_min(1.0)

    def forward(self, feature_s, logit_s, feature_t, logit_t, target):
        node_s, edge_s, graph_s = self._graph_representation(
            feature_s, logit_s, self.embed_s, self.gnn_s
        )
        node_t, edge_t, graph_t = self._graph_representation(
            feature_t, logit_t, self.embed_t, self.gnn_t
        )

        loss_node = F.mse_loss(node_s, node_t)
        loss_edge = F.mse_loss(edge_s, edge_t)
        loss_global = F.mse_loss(graph_s, graph_t)
        loss_graph = loss_node + loss_edge + loss_global
        loss_feature = self._discriminative_loss(graph_t, graph_s, target)
        return loss_graph, loss_feature
