"""
한국 주식 자동매매 - 기술적 분석 모듈
주가 데이터에 대한 기술적 분석 기능 제공
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

from korea_stock_auto.api import KoreaInvestmentApiClient
from korea_stock_auto.utils.utils import send_message
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
        # 각 심볼별, 인터벌별 데이터를 저장하는 딕셔너리
        self.symbol_data_cache: Dict[str, Dict[str, Dict[str, Any]]] = {} 
        # 예시: {'005930': {'D': {'ohlcv': df, 'rsi': rsi_series, 'macd': macd_df, ...}}}

    def _fetch_ohlcv(self, symbol: str, interval: str = 'D', limit: int = 200) -> Optional[pd.DataFrame]:
        """
        지정된 종목 및 간격으로 OHLCV 데이터를 가져옵니다.
        interval: 'D' (일봉), 'W' (주봉), 'M' (월봉), 숫자 (분봉, 예: '1', '5', '60')
        """
        logger.debug(f"Fetching OHLCV for {symbol}, interval {interval}, limit {limit}")
        try:
            # 한국투자증권 API는 일/주/월봉과 분봉 API가 다름
            if interval in ['D', 'W', 'M']: # 일봉, 주봉, 월봉
                # 일봉 데이터 조회 API 호출
                # df = self.api.get_daily_ohlcv_data(symbol, interval=interval, limit=limit) # 가정
                # df = load_sample_data() # 임시 테스트용
                # return df.sort_index()
                
                # ---- 실제 API 호출 예시 (korea_investment_api.py 에 유사 함수 필요) ----
                ohlcv_list = self.api.get_daily_stock_chart_data(symbol, period_div_code=interval.upper(), limit=limit)
                if not ohlcv_list:
                    logger.warning(f"No OHLCV data fetched for {symbol} interval {interval}.")
                    return None

                df = pd.DataFrame(ohlcv_list)
                if df.empty:
                    return None
                
                # API 응답에 따라 컬럼명 매핑 (예시, 실제 API 응답 확인 후 조정)
                # 'stck_bsop_date' -> 'date'
                # 'stck_oprc' -> 'open'
                # 'stck_hgpr' -> 'high'
                # 'stck_lwpr' -> 'low'
                # 'stck_clpr' -> 'close'
                # 'acml_vol' -> 'volume'
                rename_map = {
                    'stck_bsop_date': 'date', 'stck_oprc': 'open', 'stck_hgpr': 'high',
                    'stck_lwpr': 'low', 'stck_clpr': 'close', 'acml_vol': 'volume',
                    # 분봉의 경우 필드명이 다를 수 있음: 'stck_cntg_hour', 'stck_prpr', 'cntg_vol' 등
                    'stck_cntg_hour': 'time', # 분봉 시간
                }
                df.rename(columns=rename_map, inplace=True)

                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                elif 'time' in df.columns and interval.isdigit(): # 분봉 데이터 처리
                    # 분봉의 경우 날짜와 시간을 조합해야 할 수 있음. API 응답 형식에 따라 처리.
                    # 여기서는 'time' 컬럼이 'HHMMSS' 형식이라고 가정
                    # 실제로는 날짜 정보도 함께 받아와야 완전한 datetime 인덱스 생성 가능
                    # df['datetime'] = pd.to_datetime(df['date_from_api_or_today'] + ' ' + df['time'])
                    # df.set_index('datetime', inplace=True)
                    pass # 분봉 datetime 처리 필요

                # 필요한 컬럼만 선택하고 숫자형으로 변환
                ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in ohlcv_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    else: # API 응답에 해당 컬럼이 없을 경우
                        logger.warning(f"Column {col} not found in OHLCV data for {symbol}")
                        df[col] = 0 # 또는 다른 적절한 값으로 채우거나, 오류 처리
                
                df.dropna(subset=ohlcv_cols, inplace=True) # 필수 컬럼에 NaN이 있으면 해당 행 제거
                return df.sort_index()

            elif interval.isdigit(): # 분봉 데이터
                # 분봉 데이터 조회 API 호출
                # df = self.api.get_intraday_ohlcv_data(symbol, interval=interval, limit=limit) # 가정
                # 위와 유사하게 DataFrame으로 변환 및 전처리
                logger.warning(f"Intraday OHLCV for interval '{interval}' not yet fully implemented for fetching.")
                # 임시로 None 반환
                # ---- 실제 API 호출 예시 (korea_investment_api.py 에 유사 함수 필요) ----
                ohlcv_list = self.api.get_intraday_minute_chart_data(symbol, time_interval=interval, limit_count=limit)
                if not ohlcv_list:
                    logger.warning(f"No Intraday OHLCV data fetched for {symbol} interval {interval}.")
                    return None
                df = pd.DataFrame(ohlcv_list)
                # ... (일봉과 유사한 전처리) ...
                # 분봉의 경우 시간(stck_cntg_hour)과 현재가(stck_prpr), 거래량(cntg_vol) 등을 사용
                # 시고저종이 없을 수 있으므로, 현재가를 시고저종으로 사용하거나, OHLC를 만들어야 함.
                # 여기서는 API가 OHLC를 제공한다고 가정하거나, 현재가만 사용한다고 가정.
                # 현재는 위 일봉 처리 로직에 분봉 시간 처리 부분이 일부 포함됨.
                rename_map = {
                    'stck_bsop_date': 'date', 'stck_oprc': 'open', 'stck_hgpr': 'high',
                    'stck_lwpr': 'low', 'stck_clpr': 'close', 'acml_vol': 'volume',
                    'stck_cntg_hour': 'time', # 분봉 시간
                    'stck_prpr': 'close', # 분봉에서는 현재가를 종가로 주로 사용
                    'cntg_vol': 'volume'
                }
                df.rename(columns=rename_map, inplace=True)

                if 'date' in df.columns and 'time' in df.columns: # 날짜와 시간이 모두 있는 경우
                     # 시간 형식에 따라 HHMMSS 또는 HHMM 등으로 올 수 있음
                    df['datetime_str'] = df['date'].astype(str) + df['time'].astype(str).str.zfill(6) # 예: 20231027090000
                    df['datetime'] = pd.to_datetime(df['datetime_str'], format='%Y%m%d%H%M%S')
                    df.set_index('datetime', inplace=True)
                elif 'time' in df.columns: # 시간만 있는 경우 (날짜는 당일로 가정 - API에 따라 다름)
                    # 이 경우는 API가 당일 분봉만 제공하고 날짜 정보가 없을 때
                    # df['datetime'] = pd.to_datetime(pd.Timestamp('today').strftime('%Y-%m-%d') + ' ' + df['time'].astype(str).str.zfill(6), format='%Y-%m-%d %H%M%S')
                    # df.set_index('datetime', inplace=True)
                    logger.warning(" 분봉 데이터에 날짜 정보가 없어 정확한 datetime 인덱스 생성이 어려울 수 있습니다.")
                    # 임시로 시간 인덱스만 사용하거나, 오류 처리 필요
                    df.set_index('time', inplace=True) # 임시

                ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in ohlcv_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    else:
                        if col == 'close' and 'stck_prpr' in df.columns: # 현재가를 종가로
                             df['close'] = pd.to_numeric(df['stck_prpr'], errors='coerce')
                        elif col == 'volume' and 'cntg_vol' in df.columns:
                             df['volume'] = pd.to_numeric(df['cntg_vol'], errors='coerce')
                        else:
                            logger.warning(f"Column {col} not found in Intraday OHLCV data for {symbol}")
                            df[col] = 0
                # 분봉에서 시고저가 없는 경우 종가로 채우기 (근사치)
                if 'open' not in df.columns and 'close' in df.columns: df['open'] = df['close']
                if 'high' not in df.columns and 'close' in df.columns: df['high'] = df['close']
                if 'low' not in df.columns and 'close' in df.columns: df['low'] = df['close']
                
                df.dropna(subset=ohlcv_cols, inplace=True)
                return df.sort_index()
            else:
                logger.error(f"Unsupported interval type: {interval}")
                return None
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol} interval {interval}: {e}", exc_info=True)
            return None

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
        
        # 기술적 지표 계산 및 캐시 저장 (필요에 따라)
        # 예: self.calculate_rsi(symbol, interval)
        #     self.calculate_macd(symbol, interval)
        #     self.calculate_moving_averages(symbol, interval)
        # Trader에서 각 get_xxx 메소드를 호출할 때 내부적으로 계산하도록 변경할 수도 있음.
        # 여기서는 update_symbol_data 호출 시 주요 지표를 미리 계산하여 캐시한다고 가정.
        
        # 주요 지표들을 미리 계산해서 캐시에 넣어두자.
        self.get_rsi_values(symbol, interval)
        self.get_macd(symbol, interval)
        self.get_moving_average(symbol, interval, window=5) # 예시: 단기 이평선
        self.get_moving_average(symbol, interval, window=20) # 예시: 장기 이평선

        logger.info(f"Data and indicators updated for {symbol} interval {interval}")

    def _calculate_and_cache_indicator(self, symbol: str, interval: str, indicator_name: str, calculation_func, *args, **kwargs) -> Any:
        """
        지표를 계산하고 캐시에 저장한 뒤 결과를 반환하는 헬퍼 함수.
        indicator_name: 캐시에 저장될 지표의 키 이름 (예: 'rsi', 'macd_data')
        calculation_func: 실제 지표를 계산하는 함수 (예: talib.RSI)
        *args, **kwargs: calculation_func에 전달될 인자들
        """
        if symbol not in self.symbol_data_cache or \
           interval not in self.symbol_data_cache[symbol] or \
           'ohlcv' not in self.symbol_data_cache[symbol][interval] or \
           self.symbol_data_cache[symbol][interval]['ohlcv'].empty:
            logger.warning(f"OHLCV data for {symbol}-{interval} not available for {indicator_name} calculation.")
            return None # 또는 적절한 기본값

        # 캐시에 이미 지표가 계산되어 있는지 확인 (선택 사항: 매번 새로 계산할 수도 있음)
        # cached_indicator = self.symbol_data_cache[symbol][interval].get(indicator_name)
        # if cached_indicator is not None:
        #    return cached_indicator

        ohlcv_df = self.symbol_data_cache[symbol][interval]['ohlcv']
        if 'close' not in ohlcv_df.columns or ohlcv_df['close'].isnull().all():
            logger.error(f"Close price data is missing or all NaN for {symbol}-{interval}.")
            return None
            
        try:
            result = calculation_func(ohlcv_df['close'], *args, **kwargs)
            # 계산된 지표를 캐시에 저장
            self.symbol_data_cache[symbol][interval][indicator_name] = result
            logger.debug(f"{indicator_name} calculated and cached for {symbol}-{interval}.")
            return result
        except Exception as e:
            logger.error(f"Error calculating {indicator_name} for {symbol}-{interval}: {e}", exc_info=True)
            return None
    

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
                logger.warning(f"MACD calculation resulted in empty or all NaN series for {symbol}-{interval}")
                return None

            # NaN이 아닌 마지막 두 값 가져오기
            valid_macd = macd.dropna()
            valid_signal = macdsignal.dropna()
            valid_hist = macdhist.dropna()

            if len(valid_macd) < 2 or len(valid_signal) < 2 or len(valid_hist) < 2:
                logger.warning(f"Not enough valid (non-NaN) MACD data points for {symbol}-{interval} to get current and previous values.")
                return None
            
            result = {
                'macd': valid_macd.iloc[-1],
                'signal': valid_signal.iloc[-1],
                'histogram': valid_hist.iloc[-1],
                'prev_macd': valid_macd.iloc[-2],
                'prev_signal': valid_signal.iloc[-2],
                'prev_histogram': valid_hist.iloc[-2],
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
        if cached_ma is not None: # 캐시에 float 값으로 저장되어 있다고 가정
            return cached_ma

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
            
            last_ma = ma.dropna().iloc[-1] if not ma.dropna().empty else None
            
            if last_ma is not None:
                if symbol not in self.symbol_data_cache: self.symbol_data_cache[symbol] = {}
                if interval not in self.symbol_data_cache[symbol]: self.symbol_data_cache[symbol][interval] = {}
                self.symbol_data_cache[symbol][interval][indicator_key] = last_ma
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
                'current_rsi': valid_rsi.iloc[-1],
                'prev_rsi': valid_rsi.iloc[-2]
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
            volatility = close_prices.rolling(window=window).std().iloc[-1]
            
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
                'upper': valid_upper.iloc[-1],
                'middle': valid_middle.iloc[-1],
                'lower': valid_lower.iloc[-1]
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