"""
훈련 인터페이스
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Tuple
import pandas as pd
from datetime import datetime
from rl_system.core.interfaces.model_interface import ModelInterface


class TrainingInterface(ABC):
    """훈련 인터페이스"""
    
    @abstractmethod
    def setup_training(
        self,
        model: ModelInterface,
        training_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        훈련 설정
        
        Args:
            model: 훈련할 모델
            training_data: 훈련 데이터
            validation_data: 검증 데이터
            config: 훈련 설정
        """
        pass
    
    @abstractmethod
    def train(
        self,
        epochs: int,
        callbacks: Optional[List[Callable]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        모델 훈련 실행
        
        Args:
            epochs: 훈련 에폭 수
            callbacks: 콜백 함수 목록
            **kwargs: 추가 매개변수
            
        Returns:
            Dict[str, Any]: 훈련 결과
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
    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        체크포인트 저장
        
        Args:
            path: 저장 경로
            epoch: 현재 에폭
            metadata: 메타데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str) -> bool:
        """
        체크포인트 로드
        
        Args:
            path: 로드 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        pass
    
    @abstractmethod
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        훈련 히스토리 조회
        
        Returns:
            Dict[str, List[float]]: 훈련 히스토리
        """
        pass
    
    @abstractmethod
    def stop_training(self) -> None:
        """
        훈련 중지
        """
        pass
    
    @abstractmethod
    def resume_training(self) -> None:
        """
        훈련 재개
        """
        pass
    
    @abstractmethod
    def is_training(self) -> bool:
        """
        훈련 상태 확인
        
        Returns:
            bool: 훈련 중 여부
        """
        pass


class TrainingCallback(ABC):
    """훈련 콜백 인터페이스"""
    
    @abstractmethod
    def on_training_start(self, trainer: TrainingInterface) -> None:
        """
        훈련 시작 시 호출
        
        Args:
            trainer: 훈련 인터페이스
        """
        pass
    
    @abstractmethod
    def on_training_end(self, trainer: TrainingInterface) -> None:
        """
        훈련 종료 시 호출
        
        Args:
            trainer: 훈련 인터페이스
        """
        pass
    
    @abstractmethod
    def on_epoch_start(self, trainer: TrainingInterface, epoch: int) -> None:
        """
        에폭 시작 시 호출
        
        Args:
            trainer: 훈련 인터페이스
            epoch: 현재 에폭
        """
        pass
    
    @abstractmethod
    def on_epoch_end(
        self,
        trainer: TrainingInterface,
        epoch: int,
        logs: Dict[str, float]
    ) -> None:
        """
        에폭 종료 시 호출
        
        Args:
            trainer: 훈련 인터페이스
            epoch: 현재 에폭
            logs: 에폭 결과 로그
        """
        pass
    
    @abstractmethod
    def on_batch_start(
        self,
        trainer: TrainingInterface,
        batch: int
    ) -> None:
        """
        배치 시작 시 호출
        
        Args:
            trainer: 훈련 인터페이스
            batch: 현재 배치
        """
        pass
    
    @abstractmethod
    def on_batch_end(
        self,
        trainer: TrainingInterface,
        batch: int,
        logs: Dict[str, float]
    ) -> None:
        """
        배치 종료 시 호출
        
        Args:
            trainer: 훈련 인터페이스
            batch: 현재 배치
            logs: 배치 결과 로그
        """
        pass


class EvaluationMetric(ABC):
    """평가 지표 인터페이스"""
    
    @abstractmethod
    def calculate(
        self,
        predictions: List[Any],
        targets: List[Any]
    ) -> float:
        """
        지표 계산
        
        Args:
            predictions: 예측값
            targets: 실제값
            
        Returns:
            float: 지표 값
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        지표 이름 반환
        
        Returns:
            str: 지표 이름
        """
        pass
    
    @abstractmethod
    def is_better(self, current: float, best: float) -> bool:
        """
        현재 값이 최고값보다 나은지 확인
        
        Args:
            current: 현재 값
            best: 최고값
            
        Returns:
            bool: 더 나은지 여부
        """
        pass