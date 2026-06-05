from .AB import ABLoss
from .AT import Attention
from .CC import Correlation
from .FitNet import HintLoss
from .FSP import FSP
from .FT import FactorTransfer
from .KD import DistillKL
from .KDSVD import KDSVD
from .NST import NSTLoss
from .PKT import PKT
from .RKD import RKDLoss
from .SP import Similarity
from .VID import VIDLoss
from .GKDFD import GKDFDLoss

try:
    from .GNN import GNNLoss
    GNN_IMPORT_ERROR = None
except ImportError as error:
    GNNLoss = None
    GNN_IMPORT_ERROR = error


__all__ = [
    'ABLoss',
    'Attention',
    'Correlation',
    'DistillKL',
    'FactorTransfer',
    'FSP',
    'GKDFDLoss',
    'GNNLoss',
    'GNN_IMPORT_ERROR',
    'HintLoss',
    'KDSVD',
    'NSTLoss',
    'PKT',
    'RKDLoss',
    'Similarity',
    'VIDLoss',
]
