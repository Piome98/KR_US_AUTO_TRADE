"""
모델 서비스 - 모델 생명주기 관리
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import pandas as pd
from pathlib import Path

from rl_system.core.interfaces.model_interface import ModelInterface, EnsembleModelInterface
from rl_system.config.rl_config import RLConfig

logger = logging.getLogger(__name__)


class ModelService:
    """모델 생명주기 관리 서비스"""
    
    def __init__(self, config: RLConfig):
        """
        초기화
        
        Args:
            config: RL 시스템 설정
        """
        self.config = config
        self.models: Dict[str, ModelInterface] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # 모델 저장 디렉토리 설정
        self.models_dir = Path(config.models_dir)
        self.metadata_dir = self.models_dir / "metadata"
        self.performance_dir = self.models_dir / "performance"
        
        # 디렉토리 생성
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.performance_dir.mkdir(parents=True, exist_ok=True)
        
        # 기존 모델 메타데이터 로드
        self._load_existing_metadata()
    
    def register_model(self, model_id: str, model: ModelInterface, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        모델 등록
        
        Args:
            model_id: 모델 ID
            model: 모델 인스턴스
            metadata: 모델 메타데이터
            
        Returns:
            bool: 등록 성공 여부
        """
        try:
            # 모델 등록
            self.models[model_id] = model
            
            # 메타데이터 설정
            if metadata is None:
                metadata = {}
            
            default_metadata = {
                'model_id': model_id,
                'created_at': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'version': '1.0.0',
                'description': '',
                'parameters': model.get_parameters() if hasattr(model, 'get_parameters') else {},
                'performance': {}
            }
            
            # 메타데이터 병합
            full_metadata = {**default_metadata, **metadata}
            self.model_metadata[model_id] = full_metadata
            
            # 메타데이터 저장
            self._save_metadata(model_id, full_metadata)
            
            logger.info(f"Model registered: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            return False
    
    def get_model(self, model_id: str) -> Optional[ModelInterface]:
        """
        모델 조회
        
        Args:
            model_id: 모델 ID
            
        Returns:
            ModelInterface: 모델 인스턴스 또는 None
        """
        if model_id in self.models:
            return self.models[model_id]
        
        # 저장된 모델 로드 시도
        if self.load_model(model_id):
            return self.models.get(model_id)
        
        return None
    
    def save_model(self, model_id: str, overwrite: bool = False) -> bool:
        """
        모델 저장
        
        Args:
            model_id: 모델 ID
            overwrite: 덮어쓰기 허용 여부
            
        Returns:
            bool: 저장 성공 여부
        """
        if model_id not in self.models:
            logger.error(f"Model not found: {model_id}")
            return False
        
        try:
            model = self.models[model_id]
            model_path = self.models_dir / f"{model_id}.pkl"
            
            # 파일 존재 확인
            if model_path.exists() and not overwrite:
                logger.error(f"Model file already exists: {model_path}")
                return False
            
            # 모델 저장
            if hasattr(model, 'save'):
                # 모델 자체 저장 메서드 사용
                success = model.save(str(model_path))
                if not success:
                    logger.error(f"Model save method failed: {model_id}")
                    return False
            else:
                # 피클로 저장
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # 메타데이터 업데이트
            if model_id in self.model_metadata:
                self.model_metadata[model_id]['saved_at'] = datetime.now().isoformat()
                self.model_metadata[model_id]['file_path'] = str(model_path)
                self._save_metadata(model_id, self.model_metadata[model_id])
            
            logger.info(f"Model saved: {model_id} -> {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
            return False
    
    def load_model(self, model_id: str) -> bool:
        """
        모델 로드
        
        Args:
            model_id: 모델 ID
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            model_path = self.models_dir / f"{model_id}.pkl"
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # 모델 로드
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # 모델 인터페이스 확인
            if not isinstance(model, ModelInterface):
                logger.error(f"Loaded object is not a ModelInterface: {model_id}")
                return False
            
            # 모델 등록
            self.models[model_id] = model
            
            # 메타데이터 로드
            self._load_metadata(model_id)
            
            logger.info(f"Model loaded: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            return False
    
    def delete_model(self, model_id: str, delete_files: bool = True) -> bool:
        """
        모델 삭제
        
        Args:
            model_id: 모델 ID
            delete_files: 파일 삭제 여부
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            # 메모리에서 제거
            if model_id in self.models:
                del self.models[model_id]
            
            if model_id in self.model_metadata:
                del self.model_metadata[model_id]
            
            if model_id in self.performance_history:
                del self.performance_history[model_id]
            
            # 파일 삭제
            if delete_files:
                model_path = self.models_dir / f"{model_id}.pkl"
                metadata_path = self.metadata_dir / f"{model_id}.json"
                performance_path = self.performance_dir / f"{model_id}.json"
                
                for path in [model_path, metadata_path, performance_path]:
                    if path.exists():
                        path.unlink()
                        logger.debug(f"Deleted file: {path}")
            
            logger.info(f"Model deleted: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        모델 목록 조회
        
        Returns:
            List[Dict[str, Any]]: 모델 정보 목록
        """
        models_info = []
        
        for model_id, metadata in self.model_metadata.items():
            info = {
                'model_id': model_id,
                'model_type': metadata.get('model_type', 'Unknown'),
                'created_at': metadata.get('created_at', ''),
                'version': metadata.get('version', ''),
                'description': metadata.get('description', ''),
                'is_loaded': model_id in self.models,
                'performance': metadata.get('performance', {})
            }
            models_info.append(info)
        
        # 생성일 기준 정렬
        models_info.sort(key=lambda x: x['created_at'], reverse=True)
        
        return models_info
    
    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        모델 메타데이터 조회
        
        Args:
            model_id: 모델 ID
            
        Returns:
            Dict[str, Any]: 모델 메타데이터 또는 None
        """
        return self.model_metadata.get(model_id)
    
    def update_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> bool:
        """
        모델 메타데이터 업데이트
        
        Args:
            model_id: 모델 ID
            metadata: 업데이트할 메타데이터
            
        Returns:
            bool: 업데이트 성공 여부
        """
        if model_id not in self.model_metadata:
            logger.error(f"Model not found: {model_id}")
            return False
        
        try:
            # 메타데이터 병합
            self.model_metadata[model_id].update(metadata)
            self.model_metadata[model_id]['updated_at'] = datetime.now().isoformat()
            
            # 저장
            self._save_metadata(model_id, self.model_metadata[model_id])
            
            logger.info(f"Model metadata updated: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model metadata {model_id}: {e}")
            return False
    
    def add_performance_record(self, model_id: str, performance: Dict[str, Any]) -> bool:
        """
        모델 성능 기록 추가
        
        Args:
            model_id: 모델 ID
            performance: 성능 데이터
            
        Returns:
            bool: 추가 성공 여부
        """
        if model_id not in self.model_metadata:
            logger.error(f"Model not found: {model_id}")
            return False
        
        try:
            # 성능 기록 추가
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []
            
            performance_record = {
                'timestamp': datetime.now().isoformat(),
                **performance
            }
            
            self.performance_history[model_id].append(performance_record)
            
            # 최신 성능으로 메타데이터 업데이트
            self.model_metadata[model_id]['performance'] = performance_record
            self.model_metadata[model_id]['updated_at'] = datetime.now().isoformat()
            
            # 저장
            self._save_metadata(model_id, self.model_metadata[model_id])
            self._save_performance_history(model_id)
            
            logger.info(f"Performance record added: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add performance record {model_id}: {e}")
            return False
    
    def get_performance_history(self, model_id: str) -> List[Dict[str, Any]]:
        """
        모델 성능 기록 조회
        
        Args:
            model_id: 모델 ID
            
        Returns:
            List[Dict[str, Any]]: 성능 기록 목록
        """
        if model_id not in self.performance_history:
            # 파일에서 로드 시도
            self._load_performance_history(model_id)
        
        return self.performance_history.get(model_id, [])
    
    def compare_models(self, model_ids: List[str], metrics: List[str]) -> pd.DataFrame:
        """
        모델 성능 비교
        
        Args:
            model_ids: 비교할 모델 ID 목록
            metrics: 비교할 메트릭 목록
            
        Returns:
            pd.DataFrame: 비교 결과
        """
        comparison_data = []
        
        for model_id in model_ids:
            if model_id in self.model_metadata:
                metadata = self.model_metadata[model_id]
                performance = metadata.get('performance', {})
                
                row = {
                    'model_id': model_id,
                    'model_type': metadata.get('model_type', 'Unknown'),
                    'created_at': metadata.get('created_at', ''),
                }
                
                # 메트릭 추가
                for metric in metrics:
                    row[metric] = performance.get(metric, None)
                
                comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self, metric: str = 'total_return', higher_is_better: bool = True) -> Optional[str]:
        """
        최고 성능 모델 조회
        
        Args:
            metric: 기준 메트릭
            higher_is_better: 높을수록 좋은 메트릭 여부
            
        Returns:
            str: 최고 성능 모델 ID 또는 None
        """
        best_model = None
        best_score = float('-inf') if higher_is_better else float('inf')
        
        for model_id, metadata in self.model_metadata.items():
            performance = metadata.get('performance', {})
            if metric in performance:
                score = performance[metric]
                if isinstance(score, (int, float)):
                    if higher_is_better:
                        if score > best_score:
                            best_score = score
                            best_model = model_id
                    else:
                        if score < best_score:
                            best_score = score
                            best_model = model_id
        
        return best_model
    
    def create_ensemble(self, model_ids: List[str], weights: Optional[List[float]] = None) -> Optional[str]:
        """
        앙상블 모델 생성
        
        Args:
            model_ids: 앙상블에 포함할 모델 ID 목록
            weights: 모델 가중치 (None이면 균등 가중치)
            
        Returns:
            str: 앙상블 모델 ID 또는 None
        """
        try:
            # 모델들이 모두 로드되어 있는지 확인
            models = []
            for model_id in model_ids:
                model = self.get_model(model_id)
                if model is None:
                    logger.error(f"Model not found for ensemble: {model_id}")
                    return None
                models.append(model)
            
            # 가중치 설정
            if weights is None:
                weights = [1.0 / len(models)] * len(models)
            elif len(weights) != len(models):
                logger.error("Number of weights must match number of models")
                return None
            
            # 앙상블 모델 생성 (실제 구현은 ensemble_model.py에서)
            from rl_system.models.algorithms.ensemble_model import EnsembleModel
            ensemble = EnsembleModel(models, weights)
            
            # 앙상블 모델 등록
            ensemble_id = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            ensemble_metadata = {
                'model_type': 'EnsembleModel',
                'base_models': model_ids,
                'weights': weights,
                'description': f'Ensemble of {len(models)} models'
            }
            
            success = self.register_model(ensemble_id, ensemble, ensemble_metadata)
            if success:
                logger.info(f"Ensemble model created: {ensemble_id}")
                return ensemble_id
            else:
                return None
            
        except Exception as e:
            logger.error(f"Failed to create ensemble: {e}")
            return None
    
    def _load_existing_metadata(self) -> None:
        """기존 모델 메타데이터 로드"""
        try:
            for metadata_file in self.metadata_dir.glob("*.json"):
                model_id = metadata_file.stem
                self._load_metadata(model_id)
        except Exception as e:
            logger.error(f"Failed to load existing metadata: {e}")
    
    def _load_metadata(self, model_id: str) -> None:
        """특정 모델 메타데이터 로드"""
        try:
            metadata_path = self.metadata_dir / f"{model_id}.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.model_metadata[model_id] = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata for {model_id}: {e}")
    
    def _save_metadata(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """모델 메타데이터 저장"""
        try:
            metadata_path = self.metadata_dir / f"{model_id}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metadata for {model_id}: {e}")
    
    def _load_performance_history(self, model_id: str) -> None:
        """성능 기록 로드"""
        try:
            performance_path = self.performance_dir / f"{model_id}.json"
            if performance_path.exists():
                with open(performance_path, 'r', encoding='utf-8') as f:
                    self.performance_history[model_id] = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load performance history for {model_id}: {e}")
    
    def _save_performance_history(self, model_id: str) -> None:
        """성능 기록 저장"""
        try:
            performance_path = self.performance_dir / f"{model_id}.json"
            with open(performance_path, 'w', encoding='utf-8') as f:
                json.dump(self.performance_history[model_id], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save performance history for {model_id}: {e}")