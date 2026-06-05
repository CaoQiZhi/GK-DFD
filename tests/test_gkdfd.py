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


class GKDFDLossTest(unittest.TestCase):
    def test_forward_and_backward_are_finite(self):
        options = SimpleNamespace(
            s_dim=64,
            t_dim=128,
            feat_dim=128,
            gkdfd_cac_scale=0.5,
            gkdfd_pca_ratio=2.0,
            gkdfd_k1=7,
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
        self.assertGreater(logit_s.grad.norm().item(), 0)


if __name__ == '__main__':
    unittest.main()
