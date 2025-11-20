# GK-DFD: Learning Graph Knowledge for Discriminative Feature Distillation
## 📌 Overview
Knowledge distillation has achieved remarkable success in neural network compression and optimization. However, existing knowledge distillation methods often struggle to effectively construct the high-quality knowledge and fail to fully capture the discriminative features embedded in high-dimensional space. To address the issues, we propose a novel graph-based discriminative feature distillation framework for image classification. In graph construction, we elaborate the Unified Edge Metric and Class-Aware Connection strategy to construct an informative graph, effectively combining node-level, edge-level and global-level information. To distill more discriminative knowledge, we design the discriminative feature distillation to project high-dimensional graph representations into a low-dimensional space, and optimize the feature extraction process by minimizing intra-class distances and maximizing inter-class separability. Extensive experiments demonstrate that our method achieves competitive performance on Cifar-100 and ImageNet-1K datasets. Furthermore, comprehensive ablation studies validate the superiority of our knowledge construction method and distillation mechanism.

### Key Features:
- 🎯 **Graph Knowledge Construction**: Combines node, edge, and global embeddings via UEM and CAC
- 📊 **Unified Edge Metric**: Integrates cosine similarity and Euclidean distance for richer relation modeling
- 🔍 **Discriminative Feature Distillation**: Projects high-dimensional features into low-dimensional discriminative subspace
- 🏆 **Performance**: Validated on CIFAR-100 and ImageNet-1K

## 🚀 Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- torchvision
- scikit-learn
- numpy
- tqdm
- matplotlib (for visualization)

# The overall framework
![Overall](https://github.com/kbzhang0505/RPC/assets/97494153/f5ef7938-e72b-4b6f-899e-1bf184ad4d98)
# Knowledge Review and Preview Distillation
![Fig3](https://github.com/kbzhang0505/RPC/assets/97494153/b8c3019f-ee50-4ee0-9020-687832e2171a)
# Response Correction Mechanism
![Fig5](https://github.com/kbzhang0505/RPC/assets/97494153/21fc38c7-4c07-484f-aab1-e49aed2c5c93)

### Dataset Structure ###
For training, you need to build the new directory.

*├─data 

**└─Cifar100 

*├─logs 
### Weights ###
The weights of student models are available at https://pan.baidu.com/s/1d9CkeMjBfWEU3DPPXR8Pjg?pwd=0207.

