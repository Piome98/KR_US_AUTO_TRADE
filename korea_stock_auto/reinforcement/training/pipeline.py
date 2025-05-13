"""
한국 주식 자동매매 - 강화학습 학습 파이프라인 모듈

강화학습 모델의 전체 학습 과정을 관리하는 파이프라인 클래스 제공
"""

import os
import datetime
import logging
from typing import Dict, List, Optional, Any, Tuple, Union

from stable_baselines3.common.callbacks import CallbackList
from korea_stock_auto.utils import send_message
from korea_stock_auto.reinforcement.rl_models import RLModel, ModelEnsemble, select_features
from korea_stock_auto.reinforcement.rl_data.rl_data_manager import RLDataManager

from .base_trainer import BaseTrainer
from .evaluation import ModelEvaluator
from .callbacks import SaveOnBestTrainingRewardCallback, EarlyStoppingCallback

logger = logging.getLogger("stock_auto")

class TrainingPipeline:
    """강화학습 모델 학습 파이프라인 클래스"""
    
    def __init__(self, base_trainer: Optional[BaseTrainer] = None, evaluator: Optional[ModelEvaluator] = None):
        """
        학습 파이프라인 초기화
        
        Args:
            base_trainer (BaseTrainer): 기본 트레이너 인스턴스
            evaluator (ModelEvaluator): 모델 평가자 인스턴스
        """
        self.base_trainer = base_trainer or BaseTrainer()
        self.evaluator = evaluator or ModelEvaluator(output_dir=self.base_trainer.output_dir)
        self.data_manager = RLDataManager()
    
    def run_training_pipeline(
        self,
        code: str,
        model_type: str = "ppo",
        timesteps: int = 100000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        feature_set: str = "technical"
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        전체 학습 파이프라인 실행
        
        Args:
            code (str): 종목 코드
            model_type (str): 모델 유형 ('ppo', 'a2c', 'dqn')
            timesteps (int): 학습 스텝 수
            start_date (str): 시작 날짜
            end_date (str): 종료 날짜
            feature_set (str): 사용할 특성 세트
            
        Returns:
            tuple: (모델 ID, 평가 결과)
        """
        try:
            # 1. 데이터 로드
            send_message(f"{code} 강화학습 모델 학습 시작 ({model_type}, {timesteps}스텝)")
            logger.info(f"{code} 강화학습 모델 학습 시작 ({model_type}, {timesteps}스텝)")
            
            data = self.base_trainer.load_training_data(code, start_date, end_date)
            if data is None:
                logger.error(f"{code} 데이터 로드 실패")
                return None, None
            
            # 2. 데이터 분할
            train_data, test_data = self.base_trainer._split_data(data)
            if train_data is None or test_data is None:
                logger.error(f"{code} 데이터 분할 실패")
                return None, None
                
            # 3. 특성 선택 및 데이터 전처리
            train_data = self._preprocess_data(train_data, feature_set)
            test_data = self._preprocess_data(test_data, feature_set)
            
            # 4. 학습 환경 생성
            env = self.base_trainer.create_environment(train_data)
            if env is None:
                logger.error(f"{code} 학습 환경 생성 실패")
                return None, None
            
            # 5. 모델 학습
            model = self._train_model(env, model_type, timesteps)
            if model is None:
                logger.error(f"{code} 모델 학습 실패")
                return None, None
            
            # 6. 모델 평가
            eval_results = self._evaluate_model(model, test_data)
            
            # 7. 모델 저장
            model_id = self._save_trained_model(model, code, model_type, eval_results, feature_set)
            
            # 8. 결과 반환
            send_message(f"{code} 모델 학습 완료 (ID: {model_id}, 수익률: {eval_results['metrics']['total_return']:.2%})")
            logger.info(f"{code} 모델 학습 완료 (ID: {model_id})")
            
            return model_id, eval_results
            
        except Exception as e:
            logger.error(f"{code} 학습 파이프라인 실패: {e}", exc_info=True)
            send_message(f"{code} 학습 파이프라인 실패: {e}")
            return None, None
    
    def _preprocess_data(self, data, feature_set):
        """
        데이터 전처리 및 특성 선택
        
        Args:
            data (pd.DataFrame): 원본 데이터
            feature_set (str): 특성 세트
            
        Returns:
            pd.DataFrame: 전처리된 데이터
        """
        try:
            # 기술적 지표 추가
            processed_data = self.base_trainer.tech_indicator.add_all_indicators(data)
            
            # 데이터 정규화
            normalized_data = self.base_trainer.data_normalizer.normalize_data(processed_data)
            
            # 특성 선택
            selected_data = select_features(normalized_data, feature_set)
            
            logger.info(f"데이터 전처리 완료: 최종 특성 {len(selected_data.columns)}개")
            return selected_data
            
        except Exception as e:
            logger.error(f"데이터 전처리 실패: {e}", exc_info=True)
            return data
    
    def _train_model(self, env, model_type="ppo", timesteps=100000):
        """
        모델 학습 실행
        
        Args:
            env: 강화학습 환경
            model_type (str): 모델 유형
            timesteps (int): 학습 스텝 수
            
        Returns:
            RLModel: 학습된 모델
        """
        try:
            # 모델 ID 생성
            model_id = self.base_trainer._generate_model_id(model_type)
            model_path = os.path.join(self.base_trainer.output_dir, f"{model_id}.zip")
            
            # 모델 하이퍼파라미터 설정
            model_params = self._get_model_hyperparams(model_type)
            
            # 모델 인스턴스 생성
            model = RLModel(model_type=model_type, model_kwargs=model_params, model_path=model_path)
            
            # 콜백 설정
            callbacks = []
            
            # 최고 모델 저장 콜백
            save_callback = SaveOnBestTrainingRewardCallback(
                check_freq=1000,
                save_path=model_path,
                verbose=1
            )
            callbacks.append(save_callback)
            
            # 조기 중단 콜백
            early_stopping = EarlyStoppingCallback(
                check_freq=1000,
                patience=5,
                min_delta=0.01,
                verbose=1
            )
            callbacks.append(early_stopping)
            
            # 콜백 리스트 생성
            callback_list = CallbackList(callbacks)
            
            # 모델 학습
            logger.info(f"모델 학습 시작: {model_id}, {timesteps}스텝")
            success = model.train(env, total_timesteps=timesteps, save=True)
            
            if success:
                logger.info(f"모델 학습 완료: {model_id}")
                return model
            else:
                logger.error(f"모델 학습 실패: {model_id}")
                return None
                
        except Exception as e:
            logger.error(f"모델 학습 실패: {e}", exc_info=True)
            return None
    
    def _get_model_hyperparams(self, model_type):
        """
        모델 유형에 따른 하이퍼파라미터 반환
        
        Args:
            model_type (str): 모델 유형
            
        Returns:
            dict: 하이퍼파라미터
        """
        if model_type == "ppo":
            return {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5
            }
        elif model_type == "a2c":
            return {
                "learning_rate": 7e-4,
                "n_steps": 5,
                "gamma": 0.99,
                "gae_lambda": 1.0,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "rms_prop_eps": 1e-5
            }
        elif model_type == "dqn":
            return {
                "learning_rate": 1e-4,
                "buffer_size": 50000,
                "learning_starts": 1000,
                "batch_size": 32,
                "gamma": 0.99,
                "target_update_interval": 500,
                "train_freq": 4,
                "gradient_steps": 1,
                "exploration_fraction": 0.1,
                "exploration_final_eps": 0.05
            }
        else:
            logger.warning(f"알 수 없는 모델 유형: {model_type}, 기본 파라미터 사용")
            return {}
    
    def _evaluate_model(self, model, test_data):
        """
        모델 평가 실행
        
        Args:
            model (RLModel): 평가할 모델
            test_data (pd.DataFrame): 테스트 데이터
            
        Returns:
            dict: 평가 결과
        """
        try:
            # 모델 평가
            logger.info(f"모델 평가 시작")
            eval_results = self.evaluator.evaluate_model(
                model=model,
                test_data=test_data,
                output_prefix=os.path.basename(model.model_path).split('.')[0]
            )
            
            if eval_results:
                logger.info(f"모델 평가 완료: 수익률 {eval_results['metrics']['total_return']:.2%}")
                return eval_results
            else:
                logger.error(f"모델 평가 실패")
                return {"metrics": {"total_return": 0}}
                
        except Exception as e:
            logger.error(f"모델 평가 실패: {e}", exc_info=True)
            return {"metrics": {"total_return": 0}}
    
    def _save_trained_model(self, model, code, model_type, eval_results, feature_set):
        """
        학습된 모델 저장
        
        Args:
            model (RLModel): 저장할 모델
            code (str): 종목 코드
            model_type (str): 모델 유형
            eval_results (dict): 평가 결과
            feature_set (str): 사용한 특성 세트
            
        Returns:
            str: 모델 ID
        """
        try:
            model_id = os.path.basename(model.model_path).split('.')[0]
            
            # 모델 정보 저장
            self.base_trainer.save_model_info(
                model_id=model_id,
                model_type=model_type,
                hyperparams=self._get_model_hyperparams(model_type),
                train_period={
                    "code": code,
                    "feature_set": feature_set
                },
                results=eval_results.get("metrics", {})
            )
            
            logger.info(f"모델 저장 완료: {model_id}")
            return model_id
            
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}", exc_info=True)
            return f"{model_type}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

    def create_ensemble_from_best_models(self, top_n: int = 3) -> Optional[ModelEnsemble]:
        """
        최고 성능 모델로 앙상블 생성
        
        Args:
            top_n (int): 사용할 최고 모델 수
            
        Returns:
            ModelEnsemble: 생성된 앙상블 모델
        """
        try:
            # 모델 메타데이터 파일 찾기
            metadata_files = [f for f in os.listdir(self.base_trainer.output_dir) 
                            if f.endswith('_metadata.json')]
            
            if not metadata_files:
                logger.error("사용 가능한 모델이 없습니다.")
                return None
                
            # 모델 성능 정보 로드
            import json
            models_info = []
            
            for file in metadata_files:
                try:
                    with open(os.path.join(self.base_trainer.output_dir, file), 'r') as f:
                        metadata = json.load(f)
                    
                    model_id = metadata.get('model_id')
                    model_type = metadata.get('model_type', 'ppo')
                    total_return = metadata.get('evaluation_results', {}).get('total_return', 0)
                    sharpe_ratio = metadata.get('evaluation_results', {}).get('sharpe_ratio', 0)
                    
                    # 모델 파일 경로
                    model_path = os.path.join(self.base_trainer.output_dir, f"{model_id}.zip")
                    
                    if os.path.exists(model_path):
                        models_info.append({
                            'model_id': model_id,
                            'model_type': model_type,
                            'model_path': model_path,
                            'total_return': total_return,
                            'sharpe_ratio': sharpe_ratio,
                            'score': total_return * 0.7 + sharpe_ratio * 0.3  # 성능 점수
                        })
                except Exception as e:
                    logger.error(f"메타데이터 로드 실패 ({file}): {e}")
            
            if not models_info:
                logger.error("로드 가능한 모델이 없습니다.")
                return None
                
            # 성능 점수로 정렬
            models_info.sort(key=lambda x: x['score'], reverse=True)
            
            # 상위 N개 선택
            best_models_info = models_info[:top_n]
            
            # 모델 로드
            models = []
            for info in best_models_info:
                model = RLModel.load(info['model_path'], model_type=info['model_type'])
                if model:
                    models.append(model)
                    logger.info(f"앙상블용 모델 로드: {info['model_id']} (수익률: {info['total_return']:.2%})")
            
            # 앙상블 생성
            if len(models) > 0:
                # 성능에 비례한 가중치 설정
                weights = [info['score'] for info in best_models_info[:len(models)]]
                
                # 앙상블 인스턴스 생성
                ensemble = ModelEnsemble(models=models, weights=weights)
                
                # 앙상블 저장
                ensemble_id = f"ensemble_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                ensemble_path = os.path.join(self.base_trainer.output_dir, f"{ensemble_id}.pkl")
                ensemble.save_ensemble(ensemble_path)
                
                logger.info(f"앙상블 모델 생성 완료: {ensemble_id} ({len(models)}개 모델 사용)")
                return ensemble
            else:
                logger.error("앙상블 생성 실패: 로드된 모델이 없습니다.")
                return None
                
        except Exception as e:
            logger.error(f"앙상블 생성 실패: {e}", exc_info=True)
            return None 