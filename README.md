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
![Framework.png](https://github.com/CaoQiZhi/GK-DFD/blob/main/Fig/Framework.png)
The blue shapes represent the anchor samples, the green ones represent samples of the same class as the anchor, and the gray represent samples in class different from the anchor.
# Graph Knowledge 
<img src="https://github.com/CaoQiZhi/GK-DFD/blob/main/Fig/Graph.png" alt="Graph" width="600" height="350">

Different vertices and edges possess distinct embedding information while sharing consistent global embeddings within the same graph. The graph serves as a medium for transferring knowledge from the teacher model to the student model.
### Weights ###
The weights of student models are available at https://pan.baidu.com/s/1d9CkeMjBfWEU3DPPXR8Pjg?pwd=0207.

