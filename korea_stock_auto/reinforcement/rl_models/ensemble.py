"""
한국 주식 자동매매 - 모델 앙상블 모듈

여러 강화학습 모델을 결합한 앙상블 모델 제공
"""

import os
import pickle
import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from collections import Counter
from .model import RLModel

logger = logging.getLogger("stock_auto")

class ModelEnsemble:
    """여러 강화학습 모델의 앙상블 클래스"""
    
    def __init__(self, models: List[RLModel], weights: Optional[List[float]] = None):
        """
        모델 앙상블 초기화
        
        Args:
            models (list): 모델 리스트
            weights (list): 모델 가중치 리스트 (None인 경우 균등 가중치 적용)
        """
        self.models = models
        
        # 가중치 설정
        if weights is None:
            # 균등 가중치
            self.weights = [1.0 / len(models)] * len(models)
        else:
            # 정규화된 가중치
            weights_sum = sum(weights)
            self.weights = [w / weights_sum for w in weights]
            
        # 모델 수와 가중치 수 일치 확인
        if len(self.models) != len(self.weights):
            raise ValueError("모델 수와 가중치 수가 일치하지 않습니다.")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, Dict[str, Any]]:
        """
        앙상블 예측 수행
        
        Args:
            observation (numpy.ndarray): 관찰 벡터
            deterministic (bool): 결정적 액션 여부
            
        Returns:
            tuple: (최종 액션, 각 모델 예측 정보)
        """
        actions = []
        raw_predictions = []
        
        # 각 모델의 예측 수집
        for i, model in enumerate(self.models):
            try:
                action, _ = model.predict(observation, deterministic=deterministic)
                actions.append(action)
                raw_predictions.append({
                    'model_idx': i,
                    'action': int(action),
                    'weight': self.weights[i]
                })
            except Exception as e:
                logger.error(f"모델 {i} 예측 실패: {e}")
        
        if not actions:
            # 예측 실패 시 기본 액션 (관망)
            return 0, {'raw_predictions': [], 'voting_result': {}, 'weighted_votes': {}}
        
        # 가중 투표 방식으로 최종 액션 결정
        action_counts = Counter(actions)
        weighted_votes = {}
        
        for action, count in action_counts.items():
            # 해당 액션을 예측한 모델들의 가중치 합계
            action_weight = sum(self.weights[i] for i, a in enumerate(actions) if a == action)
            weighted_votes[int(action)] = action_weight
        
        # 가장 높은 가중치를 가진 액션 선택
        final_action = max(weighted_votes.items(), key=lambda x: x[1])[0]
        
        return final_action, {
            'raw_predictions': raw_predictions,
            'voting_result': {k: v for k, v in action_counts.items()},
            'weighted_votes': weighted_votes
        }
    
    def save_ensemble(self, path: str) -> None:
        """
        앙상블 모델 저장
        
        Args:
            path (str): 저장 경로
        """
        # 저장 디렉토리 생성
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 앙상블 메타데이터 (가중치, 모델 경로 등)
        metadata = {
            'model_count': len(self.models),
            'weights': self.weights,
            'model_paths': [model.model_path for model in self.models]
        }
        
        # 메타데이터 저장
        with open(path, 'wb') as f:
            pickle.dump(metadata, f)
            
        logger.info(f"앙상블 모델 저장 완료: {path}")
    
    @classmethod
    def load_ensemble(cls, path: str) -> 'ModelEnsemble':
        """
        저장된 앙상블 모델 로드
        
        Args:
            path (str): 메타데이터 경로
            
        Returns:
            ModelEnsemble: 로드된 앙상블 인스턴스
        """
        try:
            # 메타데이터 로드
            with open(path, 'rb') as f:
                metadata = pickle.load(f)
            
            # 개별 모델 로드
            models = []
            for model_path in metadata['model_paths']:
                # 모델 경로에서 모델 타입 추출 (파일명 규칙: ppo_20230101_... 등)
                try:
                    model_type = os.path.basename(model_path).split('_')[0]
                except:
                    model_type = "ppo"  # 기본값
                
                # 모델 로드
                model = RLModel.load(model_path, model_type=model_type)
                if model:
                    models.append(model)
            
            # 모델이 하나도 로드되지 않은 경우
            if not models:
                raise ValueError("앙상블에 포함된 모델을 로드할 수 없습니다.")
            
            # 앙상블 인스턴스 생성
            instance = cls(models=models, weights=metadata['weights'])
            logger.info(f"앙상블 모델 로드 완료: {path}")
            
            return instance
            
        except Exception as e:
            logger.error(f"앙상블 모델 로드 실패: {e}")
            return None 