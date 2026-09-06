import unittest
from types import SimpleNamespace

import numpy as np
import torch

from dataset.cifar100 import ClassBalancedBatchSampler
from distiller_zoo import GKDFDLoss


class ClassBalancedBatchSamplerTest(unittest.TestCase):
    def test_each_batch_has_equal_examples_per_class(self):
        targets = np.repeat(np.arange(100), 20)
        sampler = ClassBalancedBatchSampler(
            targets,
            batch_size=64,
            classes_per_batch=8,
        )

        for batch_index, batch in enumerate(sampler):
            labels = targets[batch]
            _, counts = np.unique(labels, return_counts=True)
            self.assertEqual(len(batch), 64)
            self.assertEqual(len(counts), 8)
            self.assertTrue(np.all(counts == 8))
            if batch_index == 9:
                break

    def test_batch_size_must_be_divisible_by_class_count(self):
        targets = np.repeat(np.arange(4), 4)
        with self.assertRaises(ValueError):
            ClassBalancedBatchSampler(targets, batch_size=7,
                                      classes_per_batch=4)


class GKDFDLossTest(unittest.TestCase):
    def test_forward_and_backward_are_finite(self):
        options = SimpleNamespace(
            s_dim=64,
            t_dim=128,
            feat_dim=128,
            gkdfd_cac_scale=0.5,
            gkdfd_pca_ratio=2.0,
            gkdfd_k1=7,
            gkdfd_k2=56,
            gkdfd_inter_weight=-1.0,
            gkdfd_dla_dim=0,
        )
        criterion = GKDFDLoss(options)
        feature_s = torch.randn(64, 64, requires_grad=True)
        feature_t = torch.randn(64, 128)
        logit_s = torch.randn(64, 100, requires_grad=True)
        logit_t = torch.randn(64, 100)
        target = torch.arange(8).repeat_interleave(8)

        loss_graph, loss_feature = criterion(
            feature_s,
            logit_s,
            feature_t,
            logit_t,
            target,
        )
        loss = 0.7 * loss_graph + 1.2 * loss_feature
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(feature_s.grad.norm().item(), 0)
        # The paper defines one shared topology from teacher logits; student
        # logits therefore receive no gradient from GK-DFD itself.
        self.assertIsNone(logit_s.grad)

    def test_uem_cac_is_shared_positive_symmetric_topology(self):
        options = SimpleNamespace(
            s_dim=4, t_dim=4, feat_dim=8, gkdfd_cac_scale=0.5,
            gkdfd_pca_ratio=2.0, gkdfd_k1=2, gkdfd_k2=3,
            gkdfd_dla_dim=0,
        )
        criterion = GKDFDLoss(options)
        logits = torch.tensor([[5., 0., 0.], [5., 0., 0.], [0., 5., 0.]])
        target = torch.tensor([0, 0, 1])
        adjacency = criterion._uem_adjacency(logits, target)
        self.assertTrue(torch.equal(adjacency, adjacency.t()))
        self.assertTrue(torch.all(adjacency >= 0))
        # Same-class edges are not CAC-attenuated; cross-class edges are.
        self.assertGreater(adjacency[0, 1], adjacency[0, 2])
        self.assertTrue(torch.all(adjacency.diag() > 0))

    def test_signed_alignment_selects_k1_and_k2_neighbors(self):
        options = SimpleNamespace(
            s_dim=4, t_dim=4, feat_dim=6, gkdfd_cac_scale=0.5,
            gkdfd_pca_ratio=1.0, gkdfd_k1=2, gkdfd_k2=3,
            gkdfd_dla_dim=0,
        )
        criterion = GKDFDLoss(options)
        target = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        reference = torch.randn(8, 6)
        selected = criterion._select_neighbors(reference, target)
        self.assertTrue(all(p.numel() == 2 and n.numel() == 3
                            for p, n in selected))
        alignment = criterion._relation_weights(reference, target)
        self.assertTrue(torch.allclose(alignment, alignment.t()))
        self.assertTrue(torch.isfinite(alignment).all())


if __name__ == '__main__':
    unittest.main()
