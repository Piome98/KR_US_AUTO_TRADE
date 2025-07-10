"""
훈련 서비스 - 훈련 파이프라인 조정
"""

import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from rl_system.core.interfaces.model_interface import ModelInterface
from rl_system.core.interfaces.training_interface import TrainingInterface, TrainingCallback
from rl_system.config.rl_config import RLConfig

logger = logging.getLogger(__name__)


class TrainingService(TrainingInterface):
    """훈련 파이프라인 조정 서비스"""
    
    def __init__(self, config: RLConfig):
        """
        초기화
        
        Args:
            config: RL 시스템 설정
        """
        self.config = config
        self.current_model: Optional[ModelInterface] = None
        self.training_data: Optional[pd.DataFrame] = None
        self.validation_data: Optional[pd.DataFrame] = None
        self.training_config: Dict[str, Any] = {}
        
        # 훈련 상태
        self.is_training_active = False
        self.current_epoch = 0
        self.training_history: Dict[str, List[float]] = {}
        self.callbacks: List[TrainingCallback] = []
        
        # 조기 종료 설정
        self.early_stopping_patience = config.get_training_config().get('early_stopping_patience', 50)
        self.best_score = float('-inf')
        self.patience_counter = 0
        
        # 체크포인트 설정
        self.checkpoint_interval = config.get_training_config().get('save_interval', 10)
        self.checkpoint_dir = config.models_dir
    
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
        self.current_model = model
        self.training_data = training_data
        self.validation_data = validation_data
        self.training_config = config or {}
        
        # 훈련 히스토리 초기화
        self.training_history = {
            'epoch': [],
            'train_reward': [],
            'train_loss': [],
            'val_reward': [],
            'val_loss': [],
            'learning_rate': [],
            'training_time': []
        }
        
        # 조기 종료 상태 초기화
        self.best_score = float('-inf')
        self.patience_counter = 0
        self.current_epoch = 0
        
        logger.info("Training setup completed")
    
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
        if self.current_model is None or self.training_data is None:
            raise ValueError("Training not properly setup. Call setup_training() first.")
        
        self.is_training_active = True
        
        # 콜백 설정
        if callbacks:
            self.callbacks = callbacks
        
        try:
            # 훈련 시작 콜백 실행
            self._execute_callbacks('on_training_start')
            
            training_start_time = time.time()
            
            for epoch in range(epochs):
                if not self.is_training_active:
                    logger.info("Training stopped by user")
                    break
                
                self.current_epoch = epoch + 1
                epoch_start_time = time.time()
                
                # 에폭 시작 콜백 실행
                self._execute_callbacks('on_epoch_start', epoch=self.current_epoch)
                
                # 에폭 훈련
                epoch_logs = self._train_epoch(epoch)
                
                # 훈련 히스토리 업데이트
                self._update_training_history(epoch_logs)
                
                # 검증 수행
                if self.validation_data is not None:
                    val_logs = self._validate_epoch(epoch)
                    epoch_logs.update(val_logs)
                
                # 에폭 시간 기록
                epoch_time = time.time() - epoch_start_time
                epoch_logs['training_time'] = epoch_time
                
                # 에폭 종료 콜백 실행
                self._execute_callbacks('on_epoch_end', epoch=self.current_epoch, logs=epoch_logs)
                
                # 조기 종료 확인
                if self._should_early_stop(epoch_logs):
                    logger.info(f"Early stopping at epoch {self.current_epoch}")
                    break
                
                # 체크포인트 저장
                if self.current_epoch % self.checkpoint_interval == 0:
                    self._save_checkpoint(self.current_epoch)
                
                # 진행상황 로깅
                self._log_epoch_progress(epoch_logs)
            
            total_training_time = time.time() - training_start_time
            
            # 훈련 종료 콜백 실행
            self._execute_callbacks('on_training_end')
            
            # 최종 결과 정리
            training_results = {
                'success': True,
                'total_epochs': self.current_epoch,
                'total_time': total_training_time,
                'best_score': self.best_score,
                'final_model': self.current_model,
                'training_history': self.training_history.copy()
            }
            
            logger.info(f"Training completed: {self.current_epoch} epochs in {total_training_time:.2f}s")
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'epoch': self.current_epoch,
                'training_history': self.training_history.copy()
            }
        finally:
            self.is_training_active = False
    
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
        if self.current_model is None:
            raise ValueError("No model available for evaluation")
        
        try:
            # 기본 메트릭 설정
            if metrics is None:
                metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            
            # 모델 평가 실행
            evaluation_results = self.current_model.evaluate(test_data, metrics)
            
            logger.info(f"Model evaluation completed: {len(metrics)} metrics")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}
    
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
        if self.current_model is None:
            logger.error("No model available for checkpoint")
            return False
        
        try:
            # 체크포인트 메타데이터 구성
            checkpoint_metadata = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'best_score': self.best_score,
                'training_history': self.training_history.copy(),
                'config': self.training_config.copy()
            }
            
            if metadata:
                checkpoint_metadata.update(metadata)
            
            # 모델 저장
            success = self.current_model.save(path, checkpoint_metadata)
            
            if success:
                logger.info(f"Checkpoint saved: epoch {epoch} -> {path}")
            else:
                logger.error(f"Failed to save checkpoint: epoch {epoch}")
            
            return success
            
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            return False
    
    def load_checkpoint(self, path: str) -> bool:
        """
        체크포인트 로드
        
        Args:
            path: 로드 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            if self.current_model is None:
                logger.error("No model available for checkpoint loading")
                return False
            
            # 모델 로드
            success = self.current_model.load(path)
            
            if success:
                logger.info(f"Checkpoint loaded: {path}")
                # 모델 정보에서 훈련 상태 복원
                model_info = self.current_model.get_model_info()
                if 'epoch' in model_info:
                    self.current_epoch = model_info['epoch']
                if 'best_score' in model_info:
                    self.best_score = model_info['best_score']
                if 'training_history' in model_info:
                    self.training_history = model_info['training_history']
            else:
                logger.error(f"Failed to load checkpoint: {path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Checkpoint load failed: {e}")
            return False
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """
        훈련 히스토리 조회
        
        Returns:
            Dict[str, List[float]]: 훈련 히스토리
        """
        return self.training_history.copy()
    
    def stop_training(self) -> None:
        """훈련 중지"""
        self.is_training_active = False
        logger.info("Training stop requested")
    
    def resume_training(self) -> None:
        """훈련 재개"""
        self.is_training_active = True
        logger.info("Training resumed")
    
    def is_training(self) -> bool:
        """
        훈련 상태 확인
        
        Returns:
            bool: 훈련 중 여부
        """
        return self.is_training_active
    
    def add_callback(self, callback: TrainingCallback) -> None:
        """
        훈련 콜백 추가
        
        Args:
            callback: 훈련 콜백
        """
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: TrainingCallback) -> None:
        """
        훈련 콜백 제거
        
        Args:
            callback: 제거할 콜백
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def hyperparameter_search(
        self,
        param_grid: Dict[str, List[Any]],
        cv_folds: int = 3,
        max_trials: int = 10
    ) -> Dict[str, Any]:
        """
        하이퍼파라미터 탐색
        
        Args:
            param_grid: 탐색할 파라미터 그리드
            cv_folds: 교차 검증 폴드 수
            max_trials: 최대 시도 횟수
            
        Returns:
            Dict[str, Any]: 최적 파라미터 및 결과
        """
        try:
            from itertools import product
            import random
            
            # 파라미터 조합 생성
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            param_combinations = list(product(*param_values))
            
            # 최대 시도 횟수만큼 랜덤 샘플링
            if len(param_combinations) > max_trials:
                param_combinations = random.sample(param_combinations, max_trials)
            
            best_params = None
            best_score = float('-inf')
            search_results = []
            
            logger.info(f"Starting hyperparameter search: {len(param_combinations)} combinations")
            
            for i, param_values in enumerate(param_combinations):
                # 파라미터 설정
                current_params = dict(zip(param_names, param_values))
                
                logger.info(f"Trial {i+1}/{len(param_combinations)}: {current_params}")
                
                # 교차 검증으로 평가
                cv_scores = self._cross_validate(current_params, cv_folds)
                avg_score = np.mean(cv_scores)
                
                # 결과 기록
                search_results.append({
                    'params': current_params.copy(),
                    'cv_scores': cv_scores,
                    'avg_score': avg_score,
                    'std_score': np.std(cv_scores)
                })
                
                # 최고 점수 업데이트
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = current_params.copy()
                
                logger.info(f"Trial {i+1} score: {avg_score:.4f} ± {np.std(cv_scores):.4f}")
            
            # 결과 정리
            search_summary = {
                'best_params': best_params,
                'best_score': best_score,
                'all_results': search_results,
                'n_trials': len(param_combinations)
            }
            
            logger.info(f"Hyperparameter search completed. Best score: {best_score:.4f}")
            logger.info(f"Best params: {best_params}")
            
            return search_summary
            
        except Exception as e:
            logger.error(f"Hyperparameter search failed: {e}")
            return {}
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        에폭 훈련 실행
        
        Args:
            epoch: 현재 에폭
            
        Returns:
            Dict[str, float]: 에폭 훈련 결과
        """
        try:
            # 모델별 훈련 로직 실행
            if hasattr(self.current_model, 'train'):
                result = self.current_model.train(
                    self.training_data,
                    **self.training_config
                )
                
                if isinstance(result, dict):
                    return result
                else:
                    return {'train_reward': 0.0, 'train_loss': 0.0}
            else:
                logger.warning("Model does not have train method")
                return {'train_reward': 0.0, 'train_loss': 0.0}
                
        except Exception as e:
            logger.error(f"Epoch training failed: {e}")
            return {'train_reward': 0.0, 'train_loss': float('inf')}
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        에폭 검증 실행
        
        Args:
            epoch: 현재 에폭
            
        Returns:
            Dict[str, float]: 검증 결과
        """
        try:
            if self.validation_data is None:
                return {}
            
            # 검증 데이터로 평가
            eval_results = self.current_model.evaluate(
                self.validation_data,
                metrics=['total_return', 'sharpe_ratio']
            )
            
            return {
                'val_reward': eval_results.get('total_return', 0.0),
                'val_loss': -eval_results.get('sharpe_ratio', 0.0)  # 음수로 변환
            }
            
        except Exception as e:
            logger.error(f"Epoch validation failed: {e}")
            return {'val_reward': 0.0, 'val_loss': float('inf')}
    
    def _update_training_history(self, logs: Dict[str, float]) -> None:
        """훈련 히스토리 업데이트"""
        self.training_history['epoch'].append(self.current_epoch)
        
        for key, value in logs.items():
            if key not in self.training_history:
                self.training_history[key] = []
            self.training_history[key].append(value)
    
    def _should_early_stop(self, logs: Dict[str, float]) -> bool:
        """조기 종료 조건 확인"""
        # 검증 보상을 기준으로 조기 종료 판단
        current_score = logs.get('val_reward', logs.get('train_reward', 0.0))
        
        if current_score > self.best_score:
            self.best_score = current_score
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.early_stopping_patience
    
    def _save_checkpoint(self, epoch: int) -> None:
        """체크포인트 저장"""
        try:
            checkpoint_path = f"{self.checkpoint_dir}/checkpoint_epoch_{epoch}.pkl"
            self.save_checkpoint(checkpoint_path, epoch)
        except Exception as e:
            logger.error(f"Failed to save checkpoint at epoch {epoch}: {e}")
    
    def _log_epoch_progress(self, logs: Dict[str, float]) -> None:
        """에폭 진행상황 로깅"""
        log_message = f"Epoch {self.current_epoch:3d}"
        
        for key, value in logs.items():
            if isinstance(value, float):
                log_message += f" | {key}: {value:.4f}"
        
        logger.info(log_message)
    
    def _execute_callbacks(self, event: str, **kwargs) -> None:
        """콜백 실행"""
        for callback in self.callbacks:
            try:
                if hasattr(callback, event):
                    getattr(callback, event)(self, **kwargs)
            except Exception as e:
                logger.error(f"Callback error in {event}: {e}")
    
    def _cross_validate(self, params: Dict[str, Any], cv_folds: int) -> List[float]:
        """
        교차 검증 수행
        
        Args:
            params: 모델 파라미터
            cv_folds: 폴드 수
            
        Returns:
            List[float]: 각 폴드별 점수
        """
        try:
            # 데이터 분할
            data_size = len(self.training_data)
            fold_size = data_size // cv_folds
            scores = []
            
            for fold in range(cv_folds):
                # 폴드별 훈련/검증 데이터 분할
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < cv_folds - 1 else data_size
                
                val_data = self.training_data.iloc[start_idx:end_idx]
                train_data = pd.concat([
                    self.training_data.iloc[:start_idx],
                    self.training_data.iloc[end_idx:]
                ])
                
                # 모델 복사 및 파라미터 적용
                model_copy = self._create_model_copy(params)
                
                # 간단한 훈련 수행
                model_copy.train(train_data)
                
                # 평가
                eval_result = model_copy.evaluate(val_data, ['total_return'])
                scores.append(eval_result.get('total_return', 0.0))
            
            return scores
            
        except Exception as e:
            logger.error(f"Cross validation failed: {e}")
            return [0.0] * cv_folds
    
    def _create_model_copy(self, params: Dict[str, Any]) -> ModelInterface:
        """
        파라미터가 적용된 모델 복사본 생성
        
        Args:
            params: 적용할 파라미터
            
        Returns:
            ModelInterface: 모델 복사본
        """
        # 기본 구현: 현재 모델 타입과 동일한 새 모델 생성
        # 실제 구현에서는 모델별로 적절한 복사 로직 필요
        model_copy = type(self.current_model)()
        model_copy.update_parameters(params)
        return model_copy