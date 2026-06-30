"""Base ABCs for architectures and losses."""
from training.base.architecture import BaseRecommenderArchitecture
from training.base.loss import BaseRecommenderLoss

__all__ = ["BaseRecommenderArchitecture", "BaseRecommenderLoss"]
