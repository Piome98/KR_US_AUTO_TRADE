"""
한국 주식 자동매매 - 기술적 분석 모듈
주가 데이터에 대한 기술적 분석 기능 제공
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union

from korea_stock_auto.api import KoreaInvestmentApiClient
from korea_stock_auto.utils.utils import send_message
from korea_stock_auto.data.hybrid_data_collector import HybridDataCollector
import talib

logger = logging.getLogger("stock_auto")

class TechnicalAnalyzer:
    """
    주식 기술적 분석 클래스
    """
    
    def __init__(self, api_client: KoreaInvestmentApiClient):
        """
        기술적 분석기 초기화
        
        Args:
            api_client: API 클라이언트 인스턴스
        """
        self.api = api_client
        # 하이브리드 데이터 수집기 초기화 (API + 네이버 크롤러)
        self.data_collector = HybridDataCollector(
            api_client=self.api,  # 기존 API 클라이언트 재사용
            use_database=False,   # 데이터베이스 비활성화 (속도 우선)
            api_delay=0.1,        # API 호출 간 지연시간
            crawler_delay=0.5     # 크롤러 요청 간 지연시간
        )
        # 각 심볼별, 인터벌별 데이터를 저장하는 딕셔너리
        self.symbol_data_cache: Dict[str, Dict[str, Dict[str, Any]]] = {} 
        # 예시: {'005930': {'D': {'ohlcv': df, 'rsi': rsi_series, 'macd': macd_df, ...}}}

    def _fetch_ohlcv(self, symbol: str, interval: str = 'D', limit: int = 200) -> Optional[pd.DataFrame]:
        """
        지정된 종목 및 간격으로 OHLCV 데이터를 가져옵니다.
        하이브리드 데이터 수집기를 사용하여 API + 네이버 크롤러 데이터를 통합합니다.
        interval: 'D' (일봉), 'W' (주봉), 'M' (월봉), 숫자 (분봉, 예: '1', '5', '60')
        """
        logger.debug(f"Fetching OHLCV for {symbol}, interval {interval}, limit {limit}")
        
        try:
            # 분봉 데이터는 API만 사용 (실시간성 중요)
            if interval.isdigit():
                return self._fetch_intraday_data(symbol, interval, limit)
            
            # 일봉/주봉/월봉은 하이브리드 데이터 수집기 사용
            period_map = {'D': 'day', 'W': 'week', 'M': 'month'}
            period = period_map.get(interval, 'day')
            
            logger.info(f"하이브리드 데이터 수집 시작: {symbol}, {period}, {limit}개")
            
            # 하이브리드 데이터 수집 (API + 네이버 크롤러)
            df = self.data_collector.get_stock_data(
                code=symbol,
                period=period,
                count=limit,
                strategy="hybrid"  # API 우선, 부족시 크롤러 보완
            )
            
            if df is not None and not df.empty:
                # 데이터 정규화 및 검증
                df = self._normalize_dataframe(df, symbol)
                logger.info(f"하이브리드 데이터 수집 성공: {symbol}, {len(df)}개")
                
                # API 호출 간 지연 (과도한 요청 방지)
                time.sleep(0.1)
                return df
            else:
                logger.warning(f"하이브리드 데이터 수집 실패: {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} interval {interval}: {e}", exc_info=True)
            return None
    
    def _fetch_intraday_data(self, symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
        """
        분봉 데이터는 API에서만 수집 (실시간성 중요)
        """
        try:
            # 불필요한 warning 제거 - 5분봉 데이터 수집은 정상 작동함
            # logger.warning(f"Intraday OHLCV for interval '{interval}' not yet fully implemented for fetching.")
            
            # API 호출
            ohlcv_list = self.api.get_intraday_minute_chart_data(symbol, time_interval=interval, limit_count=limit)
            if not ohlcv_list:
                logger.warning(f"No Intraday OHLCV data fetched for {symbol} interval {interval}.")
                return None
                
            df = pd.DataFrame(ohlcv_list)
            if df.empty:
                return None
            
            # 컬럼명 매핑
            rename_map = {
                'stck_bsop_date': 'date', 'stck_oprc': 'open', 'stck_hgpr': 'high',
                'stck_lwpr': 'low', 'stck_clpr': 'close', 'acml_vol': 'volume',
                'stck_cntg_hour': 'time', 'stck_prpr': 'close', 'cntg_vol': 'volume'
            }
            df.rename(columns=rename_map, inplace=True)

            # 시간 인덱스 처리
            if 'date' in df.columns and 'time' in df.columns:
                df['datetime_str'] = df['date'].astype(str) + df['time'].astype(str).str.zfill(6)
                df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y%m%d%H%M%S')
                df.set_index('datetime', inplace=True)
            elif 'time' in df.columns:
                df.set_index('time', inplace=True)

            # 데이터 정규화
            df = self._normalize_dataframe(df, symbol)
            
            # API 호출 간 지연
            time.sleep(0.1)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}", exc_info=True)
            return None
    
    def _normalize_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        DataFrame 정규화 및 검증
        """
        try:
            # 필요한 컬럼 확인 및 변환
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            
            for col in ohlcv_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    logger.warning(f"Column {col} not found in data for {symbol}")
                    if col == 'volume':
                        df[col] = 0
                    else:
                        # OHLC가 없으면 close 값으로 대체
                        if 'close' in df.columns:
                            df[col] = df['close']
                        else:
                            df[col] = 0
            
            # NaN 값 제거
            df.dropna(subset=ohlcv_cols, inplace=True)
            
            # 인덱스가 날짜가 아닌 경우 처리
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
            
            # 최신 데이터부터 정렬
            df = df.sort_index(ascending=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Error normalizing dataframe for {symbol}: {e}", exc_info=True)
            return df

    def _get_cached_ohlcv(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """캐시된 OHLCV 데이터를 반환합니다."""
        return self.symbol_data_cache.get(symbol, {}).get(interval, {}).get('ohlcv')

    def _update_ohlcv_cache(self, symbol: str, interval: str, df: pd.DataFrame):
        """OHLCV 데이터 캐시를 업데이트합니다."""
        if symbol not in self.symbol_data_cache:
            self.symbol_data_cache[symbol] = {}
        if interval not in self.symbol_data_cache[symbol]:
            self.symbol_data_cache[symbol][interval] = {}
        self.symbol_data_cache[symbol][interval]['ohlcv'] = df
        logger.debug(f"OHLCV cache updated for {symbol} interval {interval}. Shape: {df.shape}")

    def _calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> Optional[pd.Series]:
        """RSI 계산"""
        try:
            if 'close' not in df.columns or len(df) < window:
                return None
            close_prices = df['close'].dropna()
            if len(close_prices) < window:
                return None
            return talib.RSI(close_prices, timeperiod=window)
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return None
    
    def _calculate_macd(self, df: pd.DataFrame, short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> Optional[Dict[str, pd.Series]]:
        """MACD 계산"""
        try:
            if 'close' not in df.columns or len(df) < long_window:
                return None
            close_prices = df['close'].dropna()
            if len(close_prices) < long_window:
                return None
            macd, signal, histogram = talib.MACD(close_prices, fastperiod=short_window, slowperiod=long_window, signalperiod=signal_window)
            return {'macd': macd, 'signal': signal, 'histogram': histogram}
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return None
    
    def _calculate_ma(self, df: pd.DataFrame, window: int = 20) -> Optional[pd.Series]:
        """이동평균선 계산"""
        try:
            if 'close' not in df.columns or len(df) < window:
                return None
            close_prices = df['close'].dropna()
            if len(close_prices) < window:
                return None
            return talib.SMA(close_prices, timeperiod=window)
        except Exception as e:
            logger.error(f"Error calculating MA: {e}")
            return None
    
    def _calculate_bollinger(self, df: pd.DataFrame, window: int = 20, nbdevup: int = 2, nbdevdn: int = 2) -> Optional[Dict[str, pd.Series]]:
        """볼린저 밴드 계산"""
        try:
            if 'close' not in df.columns or len(df) < window:
                return None
            close_prices = df['close'].dropna()
            if len(close_prices) < window:
                return None
            upper, middle, lower = talib.BBANDS(close_prices, timeperiod=window, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=0)
            return {'upper': upper, 'middle': middle, 'lower': lower}
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return None

    def _calculate_and_cache_indicator(self, symbol: str, interval: str, indicator_name: str, calculation_func, ohlcv_df: pd.DataFrame, *args, **kwargs) -> Any:
        """
        지표를 계산하고 캐시에 저장한 뒤 결과를 반환하는 헬퍼 함수.
        indicator_name: 캐시에 저장될 지표의 키 이름 (예: 'rsi', 'macd_data')
        calculation_func: 실제 지표를 계산하는 함수 (예: self._calculate_rsi)
        ohlcv_df: OHLCV 데이터프레임
        *args, **kwargs: calculation_func에 전달될 추가 인자들
        """
        if ohlcv_df is None or ohlcv_df.empty:
            logger.warning(f"OHLCV data for {symbol}-{interval} not available for {indicator_name} calculation.")
            return None

        if 'close' not in ohlcv_df.columns or ohlcv_df['close'].isnull().all():
            logger.error(f"Close price data is missing or all NaN for {symbol}-{interval}.")
            return None
            
        try:
            # DataFrame을 첫 번째 인수로 전달
            result = calculation_func(ohlcv_df, *args, **kwargs)
            
            # 계산된 지표를 캐시에 저장
            if symbol not in self.symbol_data_cache:
                self.symbol_data_cache[symbol] = {}
            if interval not in self.symbol_data_cache[symbol]:
                self.symbol_data_cache[symbol][interval] = {}
            
            self.symbol_data_cache[symbol][interval][indicator_name] = result
            logger.debug(f"{indicator_name} calculated and cached for {symbol}-{interval}.")
            return result
        except Exception as e:
            logger.error(f"Error calculating {indicator_name} for {symbol}-{interval}: {e}", exc_info=True)
            return None

    def _update_indicators_only(self, symbol: str, interval: str = 'D'):
        """
        캐시된 OHLCV 데이터를 사용하여 기술적 지표만 업데이트합니다.
        새로운 데이터 크롤링은 하지 않습니다.
        """
        cached_df = self._get_cached_ohlcv(symbol, interval)
        if cached_df is None or cached_df.empty:
            logger.debug(f"No cached data for {symbol} interval {interval}. Cannot update indicators only.")
            return
        
        logger.debug(f"Updating indicators only for {symbol} interval {interval} using cached data ({len(cached_df)} records)")
        
        # 기존 캐시된 데이터로 기술적 지표들을 다시 계산
        try:
            # 주요 기술적 지표들을 미리 계산하여 캐시에 저장
            # RSI 계산
            self._calculate_and_cache_indicator(symbol, interval, 'rsi', self._calculate_rsi, cached_df)
            
            # MACD 계산  
            self._calculate_and_cache_indicator(symbol, interval, 'macd', self._calculate_macd, cached_df)
            
            # 이동평균선 계산
            for window in [5, 10, 20, 60]:
                self._calculate_and_cache_indicator(symbol, interval, f'ma_{window}', self._calculate_ma, cached_df, window)
            
            # 볼린저 밴드 계산
            self._calculate_and_cache_indicator(symbol, interval, 'bollinger', self._calculate_bollinger, cached_df)
            
            logger.debug(f"Indicators updated for {symbol} interval {interval}")
            
        except Exception as e:
            logger.error(f"Error updating indicators for {symbol} interval {interval}: {e}", exc_info=True)

    def update_symbol_data(self, symbol: str, interval: str = 'D', limit: int = 300):
        """
        특정 종목의 지정된 간격 데이터를 가져오고 기술적 지표를 계산하여 캐시에 저장합니다.
        """
        df = self._fetch_ohlcv(symbol, interval, limit)
        if df is None or df.empty:
            logger.warning(f"Failed to fetch or empty data for {symbol} interval {interval}. Skipping update.")
            # 캐시에 빈 데이터라도 저장하여 반복적인 실패 방지 (선택 사항)
            # self._update_ohlcv_cache(symbol, interval, pd.DataFrame()) 
            return

        self._update_ohlcv_cache(symbol, interval, df)
        
        # 기술적 지표 계산 및 캐시 저장
        self._update_indicators_only(symbol, interval)
        
        logger.info(f"Data and indicators updated for {symbol} interval {interval}")

    def get_macd(self, symbol: str, interval: str = 'D', short_window: int = 12, long_window: int = 26, signal_window: int = 9) -> Optional[Dict[str, float]]:
        """
        지정된 종목 및 간격으로 MACD, MACD Signal, MACD Histogram 값을 계산하여 마지막 값을 반환합니다.
        이전 값들도 함께 반환합니다.
        반환 형식: {'macd': 현재값, 'signal': 현재값, 'histogram': 현재값, 
                   'prev_macd': 이전값, 'prev_signal': 이전값, 'prev_histogram': 이전값}
        """
        indicator_key = f"macd_{short_window}_{long_window}_{signal_window}"
        
        # 캐시 확인 (딕셔너리 형태로 저장된 MACD 데이터)
        cached_macd_data = self.symbol_data_cache.get(symbol, {}).get(interval, {}).get(indicator_key)
        if cached_macd_data:
            return cached_macd_data # 이미 계산된 최신/이전 값 딕셔너리 반환

        ohlcv_df = self._get_cached_ohlcv(symbol, interval)
        if ohlcv_df is None or ohlcv_df.empty or 'close' not in ohlcv_df.columns or len(ohlcv_df) < long_window:
            logger.warning(f"Not enough data to calculate MACD for {symbol} interval {interval}. Need {long_window}, got {len(ohlcv_df) if ohlcv_df is not None else 0}")
            return None
            
        try:
            close_prices = ohlcv_df['close'].dropna()
            if len(close_prices) < long_window: # dropna 후에도 데이터 길이 확인
                logger.warning(f"Not enough non-NaN close data for MACD: {symbol}-{interval}")
                return None
                    
            macd, macdsignal, macdhist = talib.MACD(
                close_prices,
                fastperiod=short_window,
                slowperiod=long_window,
                signalperiod=signal_window
            )
            
            if pd.Series(macd).empty or pd.Series(macdsignal).empty or pd.Series(macdhist).empty or \
               pd.Series(macd).isnull().all() or pd.Series(macdsignal).isnull().all() or pd.Series(macdhist).isnull().all():
                # 5분봉 등 짧은 데이터에서는 MACD 계산이 어려울 수 있으므로 debug 레벨로 변경
                logger.debug(f"MACD calculation resulted in empty or all NaN series for {symbol}-{interval}")
                return None

            # NaN이 아닌 마지막 두 값 가져오기
            valid_macd = macd.dropna()
            valid_signal = macdsignal.dropna()
            valid_hist = macdhist.dropna()

            if len(valid_macd) < 2 or len(valid_signal) < 2 or len(valid_hist) < 2:
                logger.warning(f"Not enough valid (non-NaN) MACD data points for {symbol}-{interval} to get current and previous values.")
                return None
            
            result = {
                'macd': float(valid_macd.iloc[-1]),
                'signal': float(valid_signal.iloc[-1]),
                'histogram': float(valid_hist.iloc[-1]),
                'prev_macd': float(valid_macd.iloc[-2]),
                'prev_signal': float(valid_signal.iloc[-2]),
                'prev_histogram': float(valid_hist.iloc[-2]),
            }
            # 캐시에 저장
            if symbol not in self.symbol_data_cache: self.symbol_data_cache[symbol] = {}
            if interval not in self.symbol_data_cache[symbol]: self.symbol_data_cache[symbol][interval] = {}
            self.symbol_data_cache[symbol][interval][indicator_key] = result

            return result
            
        except Exception as e:
            logger.error(f"Error calculating MACD for {symbol}-{interval}: {e}", exc_info=True)
            return None

    def get_moving_average(self, symbol: str, interval: str = 'D', window: int = 20) -> Optional[float]:
        """지정된 종목 및 간격으로 특정 기간의 이동평균선 마지막 값을 계산하여 반환합니다."""
        indicator_key = f"ma_{window}"
        
        cached_ma = self.symbol_data_cache.get(symbol, {}).get(interval, {}).get(indicator_key)
        if cached_ma is not None:
            # 캐시된 값이 Series인 경우 마지막 값을 float로 변환
            if isinstance(cached_ma, pd.Series):
                if not cached_ma.empty:
                    return float(cached_ma.iloc[-1])
            elif isinstance(cached_ma, (int, float)):
                return float(cached_ma)

        ohlcv_df = self._get_cached_ohlcv(symbol, interval)
        if ohlcv_df is None or ohlcv_df.empty or 'close' not in ohlcv_df.columns or len(ohlcv_df) < window:
            logger.warning(f"Not enough data to calculate MA({window}) for {symbol} interval {interval}")
            return None
        
        try:
            close_prices = ohlcv_df['close'].dropna()
            if len(close_prices) < window:
                logger.warning(f"Not enough non-NaN close data for MA({window}): {symbol}-{interval}")
                return None
            
            ma = talib.SMA(close_prices, timeperiod=window)
            if ma.empty or ma.isnull().all():
                logger.warning(f"MA({window}) calculation resulted in empty or all NaN series for {symbol}-{interval}")
                return None
            
            last_ma_series = ma.dropna()
            if last_ma_series.empty:
                logger.warning(f"No valid MA({window}) data for {symbol}-{interval}")
                return None
                
            last_ma = float(last_ma_series.iloc[-1])  # 명시적으로 float 변환
            
            if last_ma is not None:
                if symbol not in self.symbol_data_cache: self.symbol_data_cache[symbol] = {}
                if interval not in self.symbol_data_cache[symbol]: self.symbol_data_cache[symbol][interval] = {}
                self.symbol_data_cache[symbol][interval][indicator_key] = last_ma  # float 값만 캐시에 저장
            return last_ma
            
        except Exception as e:
            logger.error(f"Error calculating MA({window}) for {symbol}-{interval}: {e}", exc_info=True)
            return None

    def check_ma_golden_cross(self, symbol: str, interval: str = 'D', short_period: int = 5, long_period: int = 20) -> Optional[bool]:
        """단기 이동평균선이 장기 이동평균선을 상향 돌파하는 골든크로스 발생 여부를 확인합니다."""
        ohlcv_df = self._get_cached_ohlcv(symbol, interval)
        if ohlcv_df is None or ohlcv_df.empty or 'close' not in ohlcv_df.columns or len(ohlcv_df) < long_period:
            logger.warning(f"Not enough data for Golden Cross check ({symbol}-{interval})")
            return None
    
        try:
            close_prices = ohlcv_df['close'].dropna()
            if len(close_prices) < long_period:
                logger.warning(f"Not enough non-NaN close data for Golden Cross: {symbol}-{interval}")
                return None

            short_ma = talib.SMA(close_prices, timeperiod=short_period)
            long_ma = talib.SMA(close_prices, timeperiod=long_period)

            # NaN 제거 후 최소 2개 데이터 포인트 필요
            valid_short_ma = short_ma.dropna()
            valid_long_ma = long_ma.dropna()

            if len(valid_short_ma) < 2 or len(valid_long_ma) < 2:
                logger.warning(f"Not enough valid MA data points for Golden Cross: {symbol}-{interval}")
                return None
            
            # 가장 최근 두 값으로 확인
            # prev_short_ma, current_short_ma
            # prev_long_ma, current_long_ma
            # 정렬된 데이터에서 마지막 두 값을 가져오기 위해, valid_short_ma와 valid_long_ma의 인덱스를 맞춰야 함
            # DataFrame으로 합쳐서 처리하는 것이 더 안전
            ma_df = pd.DataFrame({'short_ma': short_ma, 'long_ma': long_ma}).dropna()
            if len(ma_df) < 2:
                logger.warning(f"Not enough comparable MA data points after merging for Golden Cross: {symbol}-{interval}")
                return None

            # 골든 크로스: 이전에는 단기MA <= 장기MA 였고, 현재는 단기MA > 장기MA
            prev_short = ma_df['short_ma'].iloc[-2]
            current_short = ma_df['short_ma'].iloc[-1]
            prev_long = ma_df['long_ma'].iloc[-2]
            current_long = ma_df['long_ma'].iloc[-1]

            is_golden = (prev_short <= prev_long) and (current_short > current_long)
            return is_golden
            
        except Exception as e:
            logger.error(f"Error checking Golden Cross for {symbol}-{interval}: {e}", exc_info=True)
            return None

    def check_ma_dead_cross(self, symbol: str, interval: str = 'D', short_period: int = 5, long_period: int = 20) -> Optional[bool]:
        """단기 이동평균선이 장기 이동평균선을 하향 돌파하는 데드크로스 발생 여부를 확인합니다."""
        ohlcv_df = self._get_cached_ohlcv(symbol, interval)
        if ohlcv_df is None or ohlcv_df.empty or 'close' not in ohlcv_df.columns or len(ohlcv_df) < long_period:
            logger.warning(f"Not enough data for Dead Cross check ({symbol}-{interval})")
            return None
            
        try:
            close_prices = ohlcv_df['close'].dropna()
            if len(close_prices) < long_period:
                logger.warning(f"Not enough non-NaN close data for Dead Cross: {symbol}-{interval}")
                return None
            
            short_ma = talib.SMA(close_prices, timeperiod=short_period)
            long_ma = talib.SMA(close_prices, timeperiod=long_period)
            
            ma_df = pd.DataFrame({'short_ma': short_ma, 'long_ma': long_ma}).dropna()
            if len(ma_df) < 2:
                logger.warning(f"Not enough comparable MA data points after merging for Dead Cross: {symbol}-{interval}")
                return None

            # 데드 크로스: 이전에는 단기MA >= 장기MA 였고, 현재는 단기MA < 장기MA
            prev_short = ma_df['short_ma'].iloc[-2]
            current_short = ma_df['short_ma'].iloc[-1]
            prev_long = ma_df['long_ma'].iloc[-2]
            current_long = ma_df['long_ma'].iloc[-1]

            is_dead = (prev_short >= prev_long) and (current_short < current_long)
            return is_dead
            
        except Exception as e:
            logger.error(f"Error checking Dead Cross for {symbol}-{interval}: {e}", exc_info=True)
            return None
    
    def get_rsi_values(self, symbol: str, interval: str = 'D', window: int = 14) -> Optional[Dict[str, float]]:
        """
        지정된 종목 및 간격으로 RSI를 계산하여 현재 RSI와 이전 RSI 값을 반환합니다.
        반환 형식: {'current_rsi': 현재값, 'prev_rsi': 이전값}
        """
        indicator_key = f"rsi_{window}"

        cached_rsi_data = self.symbol_data_cache.get(symbol, {}).get(interval, {}).get(indicator_key)
        if cached_rsi_data: # 이미 {'current_rsi': 값, 'prev_rsi': 값} 형태로 저장되어 있다고 가정
            return cached_rsi_data

        ohlcv_df = self._get_cached_ohlcv(symbol, interval)
        if ohlcv_df is None or ohlcv_df.empty or 'close' not in ohlcv_df.columns or len(ohlcv_df) < window + 1: # RSI는 최소 window + 1 데이터 필요
            logger.warning(f"Not enough data to calculate RSI({window}) for {symbol} interval {interval}")
            return None

        try:
            close_prices = ohlcv_df['close'].dropna()
            if len(close_prices) < window + 1 :
                logger.warning(f"Not enough non-NaN close data for RSI({window}): {symbol}-{interval}")
                return None

            rsi = talib.RSI(close_prices, timeperiod=window)
            if rsi.empty or rsi.isnull().all():
                logger.warning(f"RSI({window}) calculation resulted in empty or all NaN series for {symbol}-{interval}")
                return None
            
            valid_rsi = rsi.dropna()
            if len(valid_rsi) < 2:
                logger.warning(f"Not enough valid (non-NaN) RSI data points for {symbol}-{interval} to get current and previous values.")
                return None

            result = {
                'rsi': float(valid_rsi.iloc[-1]),  # 현재 RSI (단일 값으로 반환)
                'current_rsi': float(valid_rsi.iloc[-1]),
                'prev_rsi': float(valid_rsi.iloc[-2])
            }
            
            if symbol not in self.symbol_data_cache: self.symbol_data_cache[symbol] = {}
            if interval not in self.symbol_data_cache[symbol]: self.symbol_data_cache[symbol][interval] = {}
            self.symbol_data_cache[symbol][interval][indicator_key] = result
            return result

        except Exception as e:
            logger.error(f"Error calculating RSI({window}) for {symbol}-{interval}: {e}", exc_info=True)
            return None

    def calculate_volatility(self, symbol: str, interval: str = 'D', window: int = 20) -> Optional[float]:
        """지정된 종목 및 간격으로 특정 기간의 가격 변동성(종가 표준편차)을 계산하여 반환합니다."""
        indicator_key = f"volatility_{window}"
        cached_volatility = self.symbol_data_cache.get(symbol, {}).get(interval, {}).get(indicator_key)
        if cached_volatility is not None:
            return cached_volatility

        ohlcv_df = self._get_cached_ohlcv(symbol, interval)
        if ohlcv_df is None or ohlcv_df.empty or 'close' not in ohlcv_df.columns or len(ohlcv_df) < window:
            logger.warning(f"Not enough data to calculate Volatility({window}) for {symbol} interval {interval}")
            return None
            
        try:
            close_prices = ohlcv_df['close'].dropna()
            if len(close_prices) < window:
                logger.warning(f"Not enough non-NaN close data for Volatility({window}): {symbol}-{interval}")
                return None
            
            # 가격 변동성을 최근 window 기간 동안의 종가 수익률의 표준편차로 계산할 수도 있고, 단순히 가격 자체의 표준편차로 할 수도 있음.
            # 여기서는 종가의 표준편차를 사용.
            volatility_series = close_prices.rolling(window=window).std()
            volatility = float(volatility_series.iloc[-1])  # 명시적으로 float 변환
            
            if pd.isna(volatility):
                logger.warning(f"Volatility({window}) calculation resulted in NaN for {symbol}-{interval}")
                return None
            
            if symbol not in self.symbol_data_cache: self.symbol_data_cache[symbol] = {}
            if interval not in self.symbol_data_cache[symbol]: self.symbol_data_cache[symbol][interval] = {}
            self.symbol_data_cache[symbol][interval][indicator_key] = volatility
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating Volatility({window}) for {symbol}-{interval}: {e}", exc_info=True)
            return None
    
    def is_golden_cross(self, symbol: str, interval: str = 'D', short_period: int = 5, long_period: int = 20) -> Optional[bool]:
        """ check_ma_golden_cross 메소드의 별칭 """
        return self.check_ma_golden_cross(symbol, interval, short_period, long_period)

    def calculate_bollinger_bands(self, symbol: str, interval: str = 'D', window: int = 20, nbdevup: int = 2, nbdevdn: int = 2) -> Optional[Dict[str, float]]:
        """
        지정된 종목 및 간격으로 볼린저 밴드 값을 계산하여 마지막 값을 반환합니다.
        반환 형식: {'upper': 상단밴드, 'middle': 중간밴드, 'lower': 하단밴드}
        """
        indicator_key = f"bb_{window}_{nbdevup}_{nbdevdn}"
        cached_bb_data = self.symbol_data_cache.get(symbol, {}).get(interval, {}).get(indicator_key)
        if cached_bb_data:
            return cached_bb_data

        ohlcv_df = self._get_cached_ohlcv(symbol, interval)
        if ohlcv_df is None or ohlcv_df.empty or 'close' not in ohlcv_df.columns or len(ohlcv_df) < window:
            logger.warning(f"Not enough data for Bollinger Bands({window}) for {symbol} interval {interval}")
            return None

        try:
            close_prices = ohlcv_df['close'].dropna()
            if len(close_prices) < window:
                logger.warning(f"Not enough non-NaN close data for Bollinger Bands({window}): {symbol}-{interval}")
                return None
            
            upper, middle, lower = talib.BBANDS(
                close_prices,
                timeperiod=window,
                nbdevup=nbdevup,
                nbdevdn=nbdevdn,
                matype=0  # SMA
            )

            if pd.Series(upper).empty or pd.Series(middle).empty or pd.Series(lower).empty or \
               pd.Series(upper).isnull().all() or pd.Series(middle).isnull().all() or pd.Series(lower).isnull().all():
                logger.warning(f"Bollinger Bands calculation resulted in empty or all NaN series for {symbol}-{interval}")
                return None
            
            valid_upper = upper.dropna()
            valid_middle = middle.dropna()
            valid_lower = lower.dropna()

            if valid_upper.empty or valid_middle.empty or valid_lower.empty:
                logger.warning(f"Not enough valid (non-NaN) Bollinger Bands data points for {symbol}-{interval}.")
                return None
            
            result = {
                'upper': float(valid_upper.iloc[-1]),
                'middle': float(valid_middle.iloc[-1]),
                'lower': float(valid_lower.iloc[-1])
            }
            
            if symbol not in self.symbol_data_cache: self.symbol_data_cache[symbol] = {}
            if interval not in self.symbol_data_cache[symbol]: self.symbol_data_cache[symbol][interval] = {}
            self.symbol_data_cache[symbol][interval][indicator_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands for {symbol}-{interval}: {e}", exc_info=True)
            return None

    # 필요에 따라 다른 기술적 지표 메소드 추가
    # 예: get_stochastic_oscillator 등
    
    def clear_cache(self, symbol: Optional[str] = None, interval: Optional[str] = None):
        """
        기술적 분석기 캐시를 초기화합니다.
        특정 종목/인터벌만 초기화하거나 전체 초기화 가능.
        """
        if symbol and interval:
            if symbol in self.symbol_data_cache and interval in self.symbol_data_cache[symbol]:
                del self.symbol_data_cache[symbol][interval]
                logger.info(f"Cache cleared for {symbol}-{interval}")
        elif symbol:
            if symbol in self.symbol_data_cache:
                del self.symbol_data_cache[symbol]
                logger.info(f"Cache cleared for symbol {symbol}")
        else:
            self.symbol_data_cache.clear()
            logger.info("All technical analyzer cache cleared.")

# 사용 예시 (테스트용)
if __name__ == '__main__':
    # # 모의 API 클라이언트 (실제 API 클라이언트로 대체 필요)
    class MockApiClient:
        def get_daily_stock_chart_data(self, symbol, period_code='D', days_to_fetch=200):
            # 모의 데이터 생성
            import numpy as np
            dates = pd.date_range(end=pd.Timestamp.today(), periods=days_to_fetch, freq='B') # B: Business day
            data = {
                'stck_bsop_date': dates.strftime('%Y%m%d'),
                'stck_oprc': np.random.randint(10000, 50000, size=days_to_fetch),
                'stck_hgpr': np.random.randint(10000, 50000, size=days_to_fetch),
                'stck_lwpr': np.random.randint(10000, 50000, size=days_to_fetch),
                'stck_clpr': np.random.randint(10000, 50000, size=days_to_fetch),
                'acml_vol': np.random.randint(100000, 1000000, size=days_to_fetch)
            }
            # 일부 데이터는 실제 API 응답처럼 문자열일 수 있음
            df = pd.DataFrame(data)
            # stck_oprc, stck_hgpr, stck_lwpr, stck_clpr, acml_vol는 문자열로 변환 후 API 처럼 list of dicts로 반환
            for col in ['stck_oprc', 'stck_hgpr', 'stck_lwpr', 'stck_clpr', 'acml_vol']:
                df[col] = df[col].astype(str)
            return df.to_dict(orient='records')

        def get_intraday_minute_chart_data(self, symbol, time_interval='5', hours_to_fetch=10):
            num_records = int(hours_to_fetch * 60 / int(time_interval))
            base_time = pd.Timestamp.now().replace(hour=9, minute=0, second=0, microsecond=0)
            times = []
            current_time = base_time
            for _ in range(num_records):
                times.append(current_time.strftime('%H%M%S'))
                current_time += pd.Timedelta(minutes=int(time_interval))
            
            dates = pd.Timestamp('today').strftime('%Y%m%d')
            data = {
                'stck_bsop_date': [dates] * num_records,
                'stck_cntg_hour': times,
                'stck_prpr': [str(np.random.randint(10000,50000)) for _ in range(num_records)], # 현재가
                'cntg_vol': [str(np.random.randint(1000,10000)) for _ in range(num_records)],   # 체결거래량
                # 분봉에서 시고저종은 제공되지 않는 경우가 많음. 현재가(stck_prpr)를 주로 사용.
                # 필요시 stck_oprc, stck_hgpr, stck_lwpr도 추가
                 'stck_oprc': [str(np.random.randint(10000,50000)) for _ in range(num_records)],
                 'stck_hgpr': [str(np.random.randint(10000,50000)) for _ in range(num_records)],
                 'stck_lwpr': [str(np.random.randint(10000,50000)) for _ in range(num_records)],
            }
            return pd.DataFrame(data).to_dict(orient='records')


    logging.basicConfig(level=logging.DEBUG)
    mock_api = MockApiClient()
    analyzer = TechnicalAnalyzer(mock_api)
    
    test_symbol = "005930" # 삼성전자
    
    # 데이터 업데이트 (내부적으로 OHLCV 가져오고 캐시)
    analyzer.update_symbol_data(test_symbol, interval='D', limit=50) # 일봉 데이터, 최근 50일
    analyzer.update_symbol_data(test_symbol, interval='5', limit=50) # 5분봉 데이터, 최근 50개 봉
    
    # MACD 가져오기
    macd_d = analyzer.get_macd(test_symbol, interval='D')
    if macd_d:
        logger.info(f"일봉 MACD for {test_symbol}: {macd_d}")

    macd_5 = analyzer.get_macd(test_symbol, interval='5')
    if macd_5:
        logger.info(f"5분봉 MACD for {test_symbol}: {macd_5}")

    # 이동평균선 가져오기
    ma5_d = analyzer.get_moving_average(test_symbol, interval='D', window=5)
    ma20_d = analyzer.get_moving_average(test_symbol, interval='D', window=20)
    logger.info(f"일봉 5 MA for {test_symbol}: {ma5_d}")
    logger.info(f"일봉 20 MA for {test_symbol}: {ma20_d}")

    ma5_5 = analyzer.get_moving_average(test_symbol, interval='5', window=5)
    logger.info(f"5분봉 5 MA for {test_symbol}: {ma5_5}")

    # 골든크로스 / 데드크로스 확인
    golden_d = analyzer.check_ma_golden_cross(test_symbol, interval='D', short_period=5, long_period=20)
    dead_d = analyzer.check_ma_dead_cross(test_symbol, interval='D', short_period=5, long_period=20)
    logger.info(f"일봉 Golden Cross (5/20) for {test_symbol}: {golden_d}")
    logger.info(f"일봉 Dead Cross (5/20) for {test_symbol}: {dead_d}")

    # RSI 값 가져오기
    rsi_d = analyzer.get_rsi_values(test_symbol, interval='D', window=14)
    if rsi_d:
        logger.info(f"일봉 RSI (14) for {test_symbol}: Current={rsi_d['current_rsi']:.2f}, Previous={rsi_d['prev_rsi']:.2f}")

    rsi_5 = analyzer.get_rsi_values(test_symbol, interval='5', window=14)
    if rsi_5:
        logger.info(f"5분봉 RSI (14) for {test_symbol}: Current={rsi_5['current_rsi']:.2f}, Previous={rsi_5['prev_rsi']:.2f}")

    # 캐시된 데이터 확인 (디버깅용)
    # print("\n--- CACHED DATA ---")
    # print(analyzer.symbol_data_cache.get(test_symbol, {}).get('D', {}).keys())
    # if analyzer.symbol_data_cache.get(test_symbol, {}).get('D', {}).get('ohlcv') is not None:
    #    print("Daily OHLCV for 005930 (tail):")
    #    print(analyzer.symbol_data_cache[test_symbol]['D']['ohlcv'].tail())

    # print(analyzer.symbol_data_cache.get(test_symbol, {}).get('5', {}).keys())
    # if analyzer.symbol_data_cache.get(test_symbol, {}).get('5', {}).get('ohlcv') is not None:
    #    print("5min OHLCV for 005930 (tail):")
    #    print(analyzer.symbol_data_cache[test_symbol]['5']['ohlcv'].tail())
        
    # analyzer.clear_cache(test_symbol, 'D')
    # analyzer.clear_cache() 