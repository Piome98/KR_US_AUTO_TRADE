"""
한국 주식 자동매매 - 모델 관리 모듈
강화학습 모델 버전 관리 및 성능 평가 시스템
"""

import os
import json
import datetime
import sqlite3
import numpy as np
import pandas as pd
from copy import deepcopy
from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.utils.db_helper import connect_db, execute_query

class ModelVersionManager:
    """강화학습 모델 버전 관리 및 성능 평가 클래스"""
    
    def __init__(self, db_path="stock_data.db", models_dir="models"):
        """
        모델 관리자 초기화
        
        Args:
            db_path (str): 데이터베이스 파일 경로
            models_dir (str): 모델 저장 디렉토리
        """
        self.db_path = db_path
        self.models_dir = models_dir
        self.models = {}  # 로드된 모델들을 저장
        self.performance_history = {}  # 모델별 성능 기록
        self.current_champion = None  # 현재 최고 성능 모델
        self.ensemble_weights = {}  # 앙상블에서의 모델 가중치
        
        # 모델 디렉토리 확인 및 생성
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 모델 관련 테이블 초기화
        self._initialize_db()
        
        # 기존 모델 로드
        self._load_models()
    
    def _initialize_db(self):
        """데이터베이스에 모델 관련 테이블 생성"""
        try:
            conn = connect_db(self.db_path)
            if not conn:
                send_message("모델 데이터베이스 연결 실패", config.notification.discord_webhook_url)
                return
            
            # 모델 버전 테이블
            execute_query(conn, '''
            CREATE TABLE IF NOT EXISTS model_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id TEXT UNIQUE,
                creation_date TEXT,
                architecture TEXT,
                hyperparameters TEXT,
                training_period TEXT,
                description TEXT
            )
            ''')
            
            # 모델 성능 테이블
            execute_query(conn, '''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id TEXT,
                evaluation_date TEXT,
                period TEXT,
                total_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                status TEXT,
                FOREIGN KEY (version_id) REFERENCES model_versions (version_id)
            )
            ''')
            
            # 모델 상벌 기록 테이블
            execute_query(conn, '''
            CREATE TABLE IF NOT EXISTS model_rewards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id TEXT,
                date TEXT,
                action_type TEXT,  -- 'REWARD' 또는 'PENALTY'
                reason TEXT,
                weight_change REAL,
                capital_change REAL,
                FOREIGN KEY (version_id) REFERENCES model_versions (version_id)
            )
            ''')
            
            conn.close()
            
        except Exception as e:
            send_message(f"모델 테이블 초기화 실패: {e}", config.notification.discord_webhook_url)
    
    def _load_models(self):
        """데이터베이스에서 모델 정보 로드"""
        try:
            conn = connect_db(self.db_path)
            if not conn:
                return
            
            # 모델 버전 정보 로드
            versions_query = "SELECT version_id FROM model_versions"
            versions_result = execute_query(conn, versions_query, fetch=True)
            
            if versions_result:
                for (version_id,) in versions_result:
                    # 모델 메타데이터 로드
                    meta_query = """
                    SELECT creation_date, architecture, hyperparameters, training_period, description 
                    FROM model_versions WHERE version_id = ?
                    """
                    meta_result = execute_query(conn, meta_query, (version_id,), fetch=True, fetchall=False)
                    
                    if meta_result:
                        creation_date, architecture, hyperparameters_json, training_period, description = meta_result
                        
                        # JSON 문자열을 딕셔너리로 변환
                        hyperparameters = json.loads(hyperparameters_json)
                        
                        # 모델 메타데이터 저장
                        self.models[version_id] = {
                            'creation_date': creation_date,
                            'architecture': architecture,
                            'hyperparameters': hyperparameters,
                            'training_period': training_period,
                            'description': description,
                            'model': None  # 실제 모델은 필요할 때 로드
                        }
                        
                        # 초기 앙상블 가중치 설정 (기본값 1.0)
                        self.ensemble_weights[version_id] = 1.0
                        
                        # 성능 기록 로드
                        self._load_performance_history(version_id)
            
            # 현재 챔피언 모델 설정
            self._set_current_champion()
            
            conn.close()
            
            send_message(f"{len(self.models, config.notification.discord_webhook_url)} 개의 모델 정보를 로드했습니다. 챔피언 모델: {self.current_champion}")
            
        except Exception as e:
            send_message(f"모델 로드 실패: {e}", config.notification.discord_webhook_url)
    
    def _load_performance_history(self, version_id):
        """특정 모델의 성능 기록 로드"""
        try:
            conn = connect_db(self.db_path)
            if not conn:
                return
            
            # 성능 기록 조회
            query = '''
            SELECT evaluation_date, period, total_return, sharpe_ratio, max_drawdown, win_rate, status
            FROM model_performance
            WHERE version_id = ?
            ORDER BY evaluation_date DESC
            '''
            
            result = execute_query(conn, query, (version_id,), fetch=True, as_df=True)
            conn.close()
            
            if result is not None and not result.empty:
                # 모델별 성능 기록 저장
                self.performance_history[version_id] = {
                    'daily': {},
                    'weekly': {},
                    'monthly': {},
                    'quarterly': {}
                }
                
                # 기간별로 최신 성능 데이터 저장
                for period in ['daily', 'weekly', 'monthly', 'quarterly']:
                    period_df = result[result['period'] == period]
                    if not period_df.empty:
                        latest = period_df.iloc[0]
                        self.performance_history[version_id][period] = {
                            'date': latest['evaluation_date'],
                            'return': latest['total_return'],
                            'sharpe': latest['sharpe_ratio'],
                            'mdd': latest['max_drawdown'],
                            'win_rate': latest['win_rate'],
                            'status': latest['status']
                        }
            
        except Exception as e:
            send_message(f"성능 기록 로드 실패 ({version_id}, config.notification.discord_webhook_url): {e}")
    
    def _set_current_champion(self):
        """성능이 가장 좋은 모델을 챔피언으로 설정"""
        best_model = None
        best_score = -float('inf')
        
        for model_id in self.models:
            if model_id in self.performance_history:
                # 월간 성능이 있으면 월간 성능 기준, 없으면 다른 기간 확인
                if 'monthly' in self.performance_history[model_id] and self.performance_history[model_id]['monthly']:
                    perf = self.performance_history[model_id]['monthly']
                    
                    # 종합 점수 계산 (수익률, 샤프 비율, MDD 고려)
                    if not perf:
                        continue
                        
                    return_val = perf.get('return', 0) 
                    sharpe = perf.get('sharpe', 0)
                    mdd = perf.get('mdd', 0)
                    
                    # MDD는 작을수록 좋으므로 부호 반전
                    score = return_val * 0.5 + sharpe * 0.3 - mdd * 0.2
                    
                    if score > best_score:
                        best_score = score
                        best_model = model_id
                        
                # 월간 성능이 없으면 다른 기간 확인
                elif any(self.performance_history[model_id].values()):
                    for period in ['weekly', 'daily', 'quarterly']:
                        if period in self.performance_history[model_id] and self.performance_history[model_id][period]:
                            perf = self.performance_history[model_id][period]
                            
                            if not perf:
                                continue
                                
                            return_val = perf.get('return', 0) 
                            sharpe = perf.get('sharpe', 0)
                            mdd = perf.get('mdd', 0)
                            
                            # MDD는 작을수록 좋으므로 부호 반전
                            score = return_val * 0.5 + sharpe * 0.3 - mdd * 0.2
                            
                            if score > best_score:
                                best_score = score
                                best_model = model_id
                            
                            break
        
        self.current_champion = best_model
        
    def save_model(self, model, version_id=None, metadata=None):
        """
        모델을 저장하고 메타데이터 기록
        
        Args:
            model: 저장할 모델 객체
            version_id (str, optional): 모델 버전 ID (없으면 자동 생성)
            metadata (dict, optional): 모델 메타데이터
            
        Returns:
            str: 저장된 모델의 버전 ID
        """
        try:
            # 버전 ID 생성 (없는 경우)
            if not version_id:
                now = datetime.datetime.now()
                version_id = f"model_{now.strftime('%Y%m%d_%H%M%S')}"
            
            # 메타데이터 기본값 설정
            if not metadata:
                metadata = {}
                
            creation_date = metadata.get('creation_date', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            architecture = metadata.get('architecture', 'Unknown')
            hyperparameters = metadata.get('hyperparameters', {})
            training_period = metadata.get('training_period', 'Unknown')
            description = metadata.get('description', '자동 저장된 모델')
            
            # 모델 파일 저장
            model_path = os.path.join(self.models_dir, f"{version_id}.pkl")
            
            # 픽클로 모델 저장
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # 메타데이터 DB에 저장
            conn = connect_db(self.db_path)
            if not conn:
                raise Exception("데이터베이스 연결 실패")
                
            hyperparameters_json = json.dumps(hyperparameters)
            
            # 모델 버전 정보 저장
            execute_query(conn, '''
            INSERT OR REPLACE INTO model_versions 
            (version_id, creation_date, architecture, hyperparameters, training_period, description)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (version_id, creation_date, architecture, hyperparameters_json, training_period, description))
            
            conn.close()
            
            # 메모리에 모델 정보 저장
            self.models[version_id] = {
                'creation_date': creation_date,
                'architecture': architecture,
                'hyperparameters': hyperparameters,
                'training_period': training_period,
                'description': description,
                'model': model  # 실제 모델도 메모리에 보관
            }
            
            # 초기 앙상블 가중치 설정
            self.ensemble_weights[version_id] = 1.0
            
            send_message(f"모델 저장 완료: {version_id}", config.notification.discord_webhook_url)
            return version_id
            
        except Exception as e:
            send_message(f"모델 저장 실패: {e}", config.notification.discord_webhook_url)
            return None
    
    def load_model(self, version_id):
        """
        특정 버전의 모델을 로드
        
        Args:
            version_id (str): 모델 버전 ID
            
        Returns:
            object: 로드된 모델 객체 또는 None (실패시)
        """
        try:
            # 이미 메모리에 로드된 모델이 있으면 반환
            if version_id in self.models and self.models[version_id]['model'] is not None:
                return self.models[version_id]['model']
            
            # 모델 파일 경로
            model_path = os.path.join(self.models_dir, f"{version_id}.pkl")
            
            # 파일 존재 확인
            if not os.path.exists(model_path):
                send_message(f"모델 파일이 존재하지 않음: {model_path}", config.notification.discord_webhook_url)
                return None
            
            # 픽클로 모델 로드
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # 메모리에 모델 저장
            if version_id in self.models:
                self.models[version_id]['model'] = model
            
            send_message(f"모델 로드 완료: {version_id}", config.notification.discord_webhook_url)
            return model
            
        except Exception as e:
            send_message(f"모델 로드 실패: {e}", config.notification.discord_webhook_url)
            return None