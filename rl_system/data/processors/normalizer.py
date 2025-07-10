"""
데이터 정규화 모듈
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class DataNormalizer:
    """데이터 정규화 클래스"""
    
    def __init__(self, method: str = 'minmax'):
        """
        초기화
        
        Args:
            method: 정규화 방법 ('minmax', 'standard', 'robust', 'quantile', 'power')
        """
        self.method = method
        self.scalers: Dict[str, Any] = {}
        self.fitted_columns: List[str] = []
        self.feature_ranges: Dict[str, Tuple[float, float]] = {}
        
        # 정규화 방법별 스케일러 생성
        self.scaler_classes = {
            'minmax': MinMaxScaler,
            'standard': StandardScaler,
            'robust': RobustScaler,
            'power': PowerTransformer
        }
    
    def fit(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> 'DataNormalizer':
        """
        정규화 파라미터 학습
        
        Args:
            data: 학습 데이터
            columns: 정규화할 컬럼 목록 (None이면 수치형 컬럼 모두)
            
        Returns:
            DataNormalizer: 자기 자신
        """
        try:
            if columns is None:
                # 수치형 컬럼 자동 선택
                columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            self.fitted_columns = columns
            
            for column in columns:
                if column not in data.columns:
                    logger.warning(f"Column not found: {column}")
                    continue
                
                # 결측값 처리
                column_data = data[column].values.reshape(-1, 1)
                
                # 무한값 제거
                column_data = self._handle_infinite_values(column_data)
                
                # 스케일러 생성 및 학습
                if self.method == 'quantile':
                    scaler = self._create_quantile_scaler()
                else:
                    scaler_class = self.scaler_classes.get(self.method, MinMaxScaler)
                    scaler = scaler_class()
                
                # 결측값이 있는 경우 처리
                if np.isnan(column_data).any():
                    imputer = SimpleImputer(strategy='median')
                    column_data = imputer.fit_transform(column_data)
                    self.scalers[f"{column}_imputer"] = imputer
                
                scaler.fit(column_data)
                self.scalers[column] = scaler
                
                # 데이터 범위 저장
                self.feature_ranges[column] = (
                    float(column_data.min()),
                    float(column_data.max())
                )
                
                logger.debug(f"Fitted normalizer for column: {column}")
            
            logger.info(f"Normalizer fitted for {len(self.fitted_columns)} columns")
            return self
            
        except Exception as e:
            logger.error(f"Failed to fit normalizer: {e}")
            return self
    
    def transform(self, data: pd.DataFrame, suffix: str = '_norm') -> pd.DataFrame:
        """
        데이터 정규화 변환
        
        Args:
            data: 변환할 데이터
            suffix: 정규화된 컬럼 접미사
            
        Returns:
            pd.DataFrame: 정규화된 데이터
        """
        try:
            result = data.copy()
            
            for column in self.fitted_columns:
                if column not in data.columns:
                    logger.warning(f"Column not found in transform data: {column}")
                    continue
                
                if column not in self.scalers:
                    logger.warning(f"Scaler not found for column: {column}")
                    continue
                
                # 데이터 준비
                column_data = data[column].values.reshape(-1, 1)
                
                # 무한값 처리
                column_data = self._handle_infinite_values(column_data)
                
                # 결측값 처리
                if f"{column}_imputer" in self.scalers:
                    column_data = self.scalers[f"{column}_imputer"].transform(column_data)
                elif np.isnan(column_data).any():
                    # 학습 시 결측값이 없었지만 변환 시 있는 경우
                    column_data = np.nan_to_num(column_data, nan=0.0)
                
                # 정규화 변환
                try:
                    normalized_data = self.scalers[column].transform(column_data)
                    result[f"{column}{suffix}"] = normalized_data.flatten()
                    logger.debug(f"Transformed column: {column}")
                except Exception as e:
                    logger.error(f"Failed to transform column {column}: {e}")
                    result[f"{column}{suffix}"] = 0.0
            
            logger.info(f"Transformed {len(self.fitted_columns)} columns")
            return result
            
        except Exception as e:
            logger.error(f"Failed to transform data: {e}")
            return data
    
    def fit_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None, suffix: str = '_norm') -> pd.DataFrame:
        """
        학습 및 변환 동시 수행
        
        Args:
            data: 학습 및 변환할 데이터
            columns: 정규화할 컬럼 목록
            suffix: 정규화된 컬럼 접미사
            
        Returns:
            pd.DataFrame: 정규화된 데이터
        """
        return self.fit(data, columns).transform(data, suffix)
    
    def inverse_transform(self, data: pd.DataFrame, suffix: str = '_norm') -> pd.DataFrame:
        """
        정규화 역변환
        
        Args:
            data: 역변환할 데이터
            suffix: 정규화된 컬럼 접미사
            
        Returns:
            pd.DataFrame: 역변환된 데이터
        """
        try:
            result = data.copy()
            
            for column in self.fitted_columns:
                norm_column = f"{column}{suffix}"
                
                if norm_column not in data.columns:
                    continue
                
                if column not in self.scalers:
                    continue
                
                # 정규화된 데이터 준비
                norm_data = data[norm_column].values.reshape(-1, 1)
                
                # 역변환
                try:
                    original_data = self.scalers[column].inverse_transform(norm_data)
                    result[f"{column}_restored"] = original_data.flatten()
                    logger.debug(f"Inverse transformed column: {column}")
                except Exception as e:
                    logger.error(f"Failed to inverse transform column {column}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to inverse transform data: {e}")
            return data
    
    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """
        특성 통계 정보 조회
        
        Returns:
            Dict[str, Dict[str, float]]: 컬럼별 통계 정보
        """
        stats = {}
        
        for column in self.fitted_columns:
            if column in self.scalers and column in self.feature_ranges:
                scaler = self.scalers[column]
                min_val, max_val = self.feature_ranges[column]
                
                column_stats = {
                    'min': min_val,
                    'max': max_val,
                    'range': max_val - min_val
                }
                
                # 스케일러별 추가 정보
                if hasattr(scaler, 'mean_'):
                    column_stats['mean'] = float(scaler.mean_[0])
                if hasattr(scaler, 'scale_'):
                    column_stats['scale'] = float(scaler.scale_[0])
                if hasattr(scaler, 'center_'):
                    column_stats['center'] = float(scaler.center_[0])
                
                stats[column] = column_stats
        
        return stats
    
    def detect_outliers(
        self,
        data: pd.DataFrame,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict[str, pd.Series]:
        """
        이상치 탐지
        
        Args:
            data: 탐지할 데이터
            method: 탐지 방법 ('iqr', 'zscore', 'isolation')
            threshold: 임계값
            
        Returns:
            Dict[str, pd.Series]: 컬럼별 이상치 마스크
        """
        outliers = {}
        
        for column in self.fitted_columns:
            if column not in data.columns:
                continue
            
            try:
                if method == 'iqr':
                    outliers[column] = self._detect_outliers_iqr(data[column], threshold)
                elif method == 'zscore':
                    outliers[column] = self._detect_outliers_zscore(data[column], threshold)
                elif method == 'isolation':
                    outliers[column] = self._detect_outliers_isolation(data[column])
                else:
                    logger.warning(f"Unknown outlier detection method: {method}")
                    outliers[column] = pd.Series(False, index=data.index)
                    
            except Exception as e:
                logger.error(f"Failed to detect outliers for {column}: {e}")
                outliers[column] = pd.Series(False, index=data.index)
        
        return outliers
    
    def handle_missing_values(
        self,
        data: pd.DataFrame,
        strategy: str = 'median',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        결측값 처리
        
        Args:
            data: 처리할 데이터
            strategy: 처리 전략 ('mean', 'median', 'mode', 'forward_fill', 'backward_fill')
            columns: 처리할 컬럼 목록
            
        Returns:
            pd.DataFrame: 결측값이 처리된 데이터
        """
        try:
            result = data.copy()
            
            if columns is None:
                columns = self.fitted_columns
            
            for column in columns:
                if column not in data.columns:
                    continue
                
                if strategy in ['mean', 'median', 'most_frequent']:
                    imputer = SimpleImputer(strategy=strategy)
                    result[column] = imputer.fit_transform(result[[column]]).flatten()
                elif strategy == 'forward_fill':
                    result[column] = result[column].fillna(method='ffill')
                elif strategy == 'backward_fill':
                    result[column] = result[column].fillna(method='bfill')
                elif strategy == 'interpolate':
                    result[column] = result[column].interpolate()
                else:
                    logger.warning(f"Unknown missing value strategy: {strategy}")
            
            logger.info(f"Handled missing values for {len(columns)} columns")
            return result
            
        except Exception as e:
            logger.error(f"Failed to handle missing values: {e}")
            return data
    
    def apply_rolling_normalization(
        self,
        data: pd.DataFrame,
        window: int = 20,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        롤링 정규화 적용
        
        Args:
            data: 정규화할 데이터
            window: 롤링 윈도우 크기
            columns: 정규화할 컬럼 목록
            
        Returns:
            pd.DataFrame: 롤링 정규화된 데이터
        """
        try:
            result = data.copy()
            
            if columns is None:
                columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            for column in columns:
                if column not in data.columns:
                    continue
                
                # 롤링 평균과 표준편차 계산
                rolling_mean = data[column].rolling(window=window, min_periods=1).mean()
                rolling_std = data[column].rolling(window=window, min_periods=1).std()
                
                # Z-score 정규화
                result[f"{column}_rolling_norm"] = (data[column] - rolling_mean) / (rolling_std + 1e-8)
                
                logger.debug(f"Applied rolling normalization for: {column}")
            
            logger.info(f"Applied rolling normalization for {len(columns)} columns")
            return result
            
        except Exception as e:
            logger.error(f"Failed to apply rolling normalization: {e}")
            return data
    
    def _handle_infinite_values(self, data: np.ndarray) -> np.ndarray:
        """무한값 처리"""
        data = np.where(np.isinf(data), np.nan, data)
        return data
    
    def _create_quantile_scaler(self):
        """분위수 정규화 스케일러 생성"""
        from sklearn.preprocessing import QuantileTransformer
        return QuantileTransformer(output_distribution='uniform', random_state=42)
    
    def _detect_outliers_iqr(self, data: pd.Series, threshold: float = 1.5) -> pd.Series:
        """IQR 방법으로 이상치 탐지"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        return (data < lower_bound) | (data > upper_bound)
    
    def _detect_outliers_zscore(self, data: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Z-score 방법으로 이상치 탐지"""
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > threshold
    
    def _detect_outliers_isolation(self, data: pd.Series) -> pd.Series:
        """Isolation Forest로 이상치 탐지"""
        try:
            from sklearn.ensemble import IsolationForest
            
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = isolation_forest.fit_predict(data.values.reshape(-1, 1))
            
            return pd.Series(outliers == -1, index=data.index)
            
        except ImportError:
            logger.warning("scikit-learn not available for isolation forest")
            return pd.Series(False, index=data.index)
    
    def save_scaler(self, filepath: str) -> bool:
        """
        스케일러 저장
        
        Args:
            filepath: 저장 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            import pickle
            
            scaler_data = {
                'method': self.method,
                'scalers': self.scalers,
                'fitted_columns': self.fitted_columns,
                'feature_ranges': self.feature_ranges
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(scaler_data, f)
            
            logger.info(f"Scaler saved: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save scaler: {e}")
            return False
    
    def load_scaler(self, filepath: str) -> bool:
        """
        스케일러 로드
        
        Args:
            filepath: 로드 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                scaler_data = pickle.load(f)
            
            self.method = scaler_data['method']
            self.scalers = scaler_data['scalers']
            self.fitted_columns = scaler_data['fitted_columns']
            self.feature_ranges = scaler_data['feature_ranges']
            
            logger.info(f"Scaler loaded: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load scaler: {e}")
            return False
    
    def get_normalization_summary(self) -> Dict[str, Any]:
        """
        정규화 요약 정보 조회
        
        Returns:
            Dict[str, Any]: 정규화 요약 정보
        """
        return {
            'method': self.method,
            'fitted_columns': self.fitted_columns,
            'num_columns': len(self.fitted_columns),
            'feature_ranges': self.feature_ranges,
            'scalers_available': list(self.scalers.keys())
        }