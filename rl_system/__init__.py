"""
통합 강화학습 시스템 (RL System)
한국 주식 및 미국 주식 자동매매를 위한 통합 강화학습 플랫폼
"""

__version__ = "1.0.0"
__author__ = "Stock Trading RL System"

from rl_system.core.di_container import RLContainer
from rl_system.data.unified_data_manager import UnifiedDataManager
from rl_system.models.model_manager import ModelManager
from rl_system.training.trainers.pipeline_trainer import PipelineTrainer

__all__ = [
    'RLContainer',
    'UnifiedDataManager', 
    'ModelManager',
    'PipelineTrainer'
]