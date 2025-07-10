"""
모델 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime


class ModelInterface(ABC):
    """강화학습 모델 추상 인터페이스"""
    
    @abstractmethod
    def train(
        self,
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        모델 훈련
        
        Args:
            training_data: 훈련 데이터
            validation_data: 검증 데이터
            **kwargs: 추가 매개변수
            
        Returns:
            Dict[str, Any]: 훈련 결과
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        observation: Union[np.ndarray, pd.DataFrame],
        deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        행동 예측
        
        Args:
            observation: 관찰 상태
            deterministic: 결정론적 예측 여부
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: (행동, 상태 값)
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        test_data: pd.DataFrame,
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """
        모델 평가
        
        Args:
            test_data: 테스트 데이터
            metrics: 평가 지표 목록
            
        Returns:
            Dict[str, float]: 평가 결과
        """
        pass
    
    @abstractmethod
    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        모델 저장
        
        Args:
            path: 저장 경로
            metadata: 메타데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> bool:
        """
        모델 로드
        
        Args:
            path: 로드 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 조회
        
        Returns:
            Dict[str, Any]: 모델 정보
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        특성 중요도 조회
        
        Returns:
            Optional[Dict[str, float]]: 특성 중요도
        """
        pass
    
    @abstractmethod
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        모델 매개변수 업데이트
        
        Args:
            parameters: 업데이트할 매개변수
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        모델 매개변수 조회
        
        Returns:
            Dict[str, Any]: 모델 매개변수
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        모델 상태 초기화
        """
        pass


class EnsembleModelInterface(ModelInterface):
    """앙상블 모델 인터페이스"""
    
    @abstractmethod
    def add_model(self, model: ModelInterface, weight: float = 1.0) -> None:
        """
        모델 추가
        
        Args:
            model: 추가할 모델
            weight: 모델 가중치
        """
        pass
    
    @abstractmethod
    def remove_model(self, model_id: str) -> bool:
        """
        모델 제거
        
        Args:
            model_id: 제거할 모델 ID
            
        Returns:
            bool: 제거 성공 여부
        """
        pass
    
    @abstractmethod
    def get_models(self) -> List[Tuple[str, ModelInterface, float]]:
        """
        앙상블 모델 목록 조회
        
        Returns:
            List[Tuple[str, ModelInterface, float]]: (모델 ID, 모델, 가중치)
        """
        pass
    
    @abstractmethod
    def update_weights(self, weights: Dict[str, float]) -> None:
        """
        모델 가중치 업데이트
        
        Args:
            weights: 업데이트할 가중치
        """
        pass
    
    @abstractmethod
    def get_weights(self) -> Dict[str, float]:
        """
        모델 가중치 조회
        
        Returns:
            Dict[str, float]: 모델 가중치
        """
        pass