"""
강화학습을 위한 데이터 수집 모듈
"""

import os
import json
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from korea_stock_auto.utils import send_message
from korea_stock_auto.data.database import DatabaseManager
import logging
from typing import Optional, Dict, List, Any, Union, Tuple

logger = logging.getLogger("stock_auto")

class DataFetcher:
    """강화학습용 데이터 수집 클래스"""
    
    def __init__(self):
        """초기화"""
        # 캐시 디렉토리 설정
        self.cache_dir = os.path.join(os.path.dirname(__file__), '../../../data/cache')
        os.makedirs(self.cache_dir, exist_ok=True)
    
    # ----- 1단계: 데이터 수집 함수들 -----
    
    def update_stock_data(self, code: str, days: int = 30, save: bool = True) -> pd.DataFrame:
        """
        종목 데이터 업데이트 (DB + API 데이터 통합)
        
        Args:
            code (str): 종목 코드
            days (int): 가져올 일수
            save (bool): 파일로 저장 여부
            
        Returns:
            pd.DataFrame: 업데이트된 데이터
        """
        try:
            # 현재 날짜 및 시작 날짜 계산
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            # 날짜 형식 변환
            end_date_str = end_date.strftime('%Y-%m-%d')
            start_date_str = start_date.strftime('%Y-%m-%d')
            
            # 1. 기존 데이터 로드
            existing_data = self.fetch_from_database(code, start_date=None, end_date=start_date_str, save=False)
            
            # 2. API로 최신 데이터 가져오기
            new_data = self.fetch_from_api(code, start_date=start_date_str, end_date=end_date_str, save=False)
            
            # 3. 데이터 합치기
            if existing_data is not None and not existing_data.empty and new_data is not None and not new_data.empty:
                # 기존 데이터와 새 데이터 결합
                combined_data = pd.concat([existing_data, new_data])
                combined_data = combined_data.drop_duplicates(subset=['date']).sort_values('date')
                
                if save:
                    self.save_to_csv(combined_data, code, suffix='stock_data')
                
                logger.info(f"{code} 데이터 통합 완료: 기존 {len(existing_data)}행 + 신규 {len(new_data)}행 = 최종 {len(combined_data)}행")
                return combined_data
                
            elif new_data is not None and not new_data.empty:
                # 새 데이터만 있는 경우
                if save:
                    self.save_to_csv(new_data, code, suffix='stock_data')
                
                logger.info(f"{code} 신규 데이터만 가져옴: {len(new_data)}행")
                return new_data
                
            else:
                # 데이터가 없는 경우
                logger.warning(f"{code} 가져올 데이터가 없습니다")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"주가 데이터 업데이트 실패: {e}", exc_info=True)
            return pd.DataFrame()
    
    def fetch_from_database(self, code: str, start_date=None, end_date=None, save: bool = False) -> pd.DataFrame:
        """
        데이터베이스에서 주가 데이터 가져오기
        
        Args:
            code (str): 종목 코드
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
            save (bool): 파일로 저장 여부
            
        Returns:
            pd.DataFrame: 주가 데이터
        """
        try:
            # 캐시 파일 확인
            cache_file = os.path.join(self.cache_dir, f'{code}_stock_data.csv')
            
            if os.path.exists(cache_file):
                # 캐시 파일 읽기
                df = pd.read_csv(cache_file)
                
                # 날짜 필터링
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    
                    if start_date:
                        start_date = pd.to_datetime(start_date)
                        df = df[df['date'] >= start_date]
                        
                    if end_date:
                        end_date = pd.to_datetime(end_date)
                        df = df[df['date'] <= end_date]
                
                if df.empty:
                    logger.warning(f"{code} 캐시에서 해당 기간 데이터를 찾을 수 없습니다")
                    return pd.DataFrame()
                
                logger.info(f"{code} 캐시에서 {len(df)}개 데이터 로드 성공")
                return df
            else:
                logger.warning(f"{code} 캐시 파일이 존재하지 않습니다")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"데이터베이스에서 데이터 가져오기 실패: {e}", exc_info=True)
            return pd.DataFrame()
    
    def fetch_from_api(self, code: str, start_date=None, end_date=None, save: bool = False) -> pd.DataFrame:
        """
        API를 통해 주가 데이터 가져오기
        
        Args:
            code (str): 종목 코드
            start_date (str): 시작 날짜 (YYYY-MM-DD)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
            save (bool): 파일로 저장 여부
            
        Returns:
            pd.DataFrame: 주가 데이터
        """
        try:
            # 1. 현재가 데이터 로드
            price_data = self.load_price_data(code)
            
            # 2. 실시간 시세 데이터 로드
            realtime_data = self.load_realtime_data(code)
            
            # 데이터가 없는 경우 빈 DataFrame 반환
            if not price_data and not realtime_data:
                logger.warning(f"{code} API 캐시 데이터가 없습니다")
                return pd.DataFrame()
            
            # 3. 현재 날짜 기준으로 DataFrame 생성
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            
            # 4. 현재가 데이터와 실시간 데이터 통합
            combined_data = {
                'date': today,
                'code': code,
                'open': price_data.get('open_price', 0),
                'high': price_data.get('high_price', 0),
                'low': price_data.get('low_price', 0),
                'close': price_data.get('current_price', 0),
                'volume': price_data.get('volume', 0),
                'change_rate': price_data.get('change_rate', 0),
                'market_cap': price_data.get('market_cap', 0)
            }
            
            # 5. 실시간 데이터가 있으면 추가 특성 포함
            if realtime_data:
                combined_data.update({
                    'bid_ask_ratio': realtime_data.get('bid_ask_ratio', 0),
                    'market_pressure': realtime_data.get('market_pressure', 0),
                    'price_volatility': realtime_data.get('price_volatility', 0),
                    'highest_ask': realtime_data.get('highest_ask', 0),
                    'lowest_bid': realtime_data.get('lowest_bid', 0),
                    'spread': realtime_data.get('spread', 0),
                })
            
            # 6. DataFrame 생성
            df = pd.DataFrame([combined_data])
            
            # 7. 저장 옵션이 켜져 있으면 저장
            if save:
                self.save_to_csv(df, code, suffix='api_data')
            
            logger.info(f"{code} API 데이터 가져오기 완료")
            return df
            
        except Exception as e:
            logger.error(f"API에서 데이터 가져오기 실패: {e}", exc_info=True)
            return pd.DataFrame()
    
    def get_latest_market_data(self, code: str) -> Dict[str, Any]:
        """
        최신 시장 데이터 조회 (현재가, 호가, 거래량 등)
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict: 최신 시장 데이터
        """
        try:
            # 1. 실시간 시세 데이터 로드
            realtime_data = self.load_realtime_data(code)
            
            # 2. 현재가 데이터 로드
            price_data = self.load_price_data(code)
            
            # 3. 두 데이터 통합
            market_data = {}
            
            # 현재가 데이터 추가
            if price_data:
                market_data.update({
                    'stock_code': code,
                    'stock_name': price_data.get('stock_name', ''),
                    'market': price_data.get('market', ''),
                    'current_price': price_data.get('current_price', 0),
                    'open_price': price_data.get('open_price', 0),
                    'high_price': price_data.get('high_price', 0),
                    'low_price': price_data.get('low_price', 0),
                    'change_rate': price_data.get('change_rate', 0),
                    'volume': price_data.get('volume', 0),
                    'market_cap': price_data.get('market_cap', 0),
                })
            
            # 실시간 시세 데이터 추가
            if realtime_data:
                market_data.update({
                    'bid_ask_ratio': realtime_data.get('bid_ask_ratio', 0),
                    'market_pressure': realtime_data.get('market_pressure', 0),
                    'price_volatility': realtime_data.get('price_volatility', 0),
                    'highest_ask': realtime_data.get('highest_ask', 0),
                    'lowest_bid': realtime_data.get('lowest_bid', 0),
                    'spread': realtime_data.get('spread', 0),
                    'timestamp': realtime_data.get('timestamp', ''),
                })
            
            # 시간 정보 추가
            market_data['fetch_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 결과 없으면 빈 딕셔너리 반환
            if not market_data:
                logger.warning(f"{code} 최신 시장 데이터를 찾을 수 없습니다")
                return {}
            
            logger.info(f"{code} 최신 시장 데이터 로드 완료")
            return market_data
            
        except Exception as e:
            logger.error(f"최신 시장 데이터 조회 실패: {e}", exc_info=True)
            return {}
    
    # ----- 2단계: 데이터 관리 유틸리티 함수들 -----
    
    def load_price_data(self, code: str) -> Dict[str, Any]:
        """
        현재가 캐시 데이터 로드
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict: 현재가 데이터
        """
        price_cache = os.path.join(self.cache_dir, f'price_{code}.json')
        
        price_data = {}
        if os.path.exists(price_cache):
            try:
                with open(price_cache, 'r') as f:
                    price_data = json.load(f)
                
                # 캐시 시간 확인 (8시간 이내)
                cache_time = datetime.datetime.fromtimestamp(os.path.getmtime(price_cache))
                current_time = datetime.datetime.now()
                
                if (current_time - cache_time).total_seconds() < 28800:  # 8시간
                    logger.info(f"{code} 현재가 데이터 캐시 사용 (캐시 시간: {cache_time.strftime('%Y-%m-%d %H:%M:%S')})")
                else:
                    logger.warning(f"{code} 현재가 데이터 캐시가 오래됨 (캐시 시간: {cache_time.strftime('%Y-%m-%d %H:%M:%S')})")
            except Exception as e:
                logger.error(f"{code} 현재가 데이터 로드 실패: {e}")
        
        return price_data
    
    def load_realtime_data(self, code: str) -> Dict[str, Any]:
        """
        실시간 시세 캐시 데이터 로드
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict: 실시간 시세 데이터
        """
        realtime_cache = os.path.join(self.cache_dir, f'realtime_{code}.json')
        
        realtime_data = {}
        if os.path.exists(realtime_cache):
            try:
                with open(realtime_cache, 'r') as f:
                    realtime_data = json.load(f)
                
                # 캐시 시간 확인 (1시간 이내)
                cache_time = datetime.datetime.fromtimestamp(os.path.getmtime(realtime_cache))
                current_time = datetime.datetime.now()
                
                if (current_time - cache_time).total_seconds() < 3600:  # 1시간
                    logger.info(f"{code} 실시간 시세 캐시 사용 (캐시 시간: {cache_time.strftime('%Y-%m-%d %H:%M:%S')})")
                else:
                    logger.warning(f"{code} 실시간 시세 캐시가 오래됨 (캐시 시간: {cache_time.strftime('%Y-%m-%d %H:%M:%S')})")
            except Exception as e:
                logger.error(f"{code} 실시간 시세 데이터 로드 실패: {e}")
        
        return realtime_data
    
    def save_to_csv(self, df: pd.DataFrame, code: str, suffix: str = 'data') -> str:
        """
        DataFrame을 CSV 파일로 저장
        
        Args:
            df (pd.DataFrame): 저장할 데이터
            code (str): 종목 코드
            suffix (str): 파일명 접미사
            
        Returns:
            str: 저장된 파일 경로
        """
        if df.empty:
            logger.warning(f"{code} 저장할 데이터가 없습니다")
            return ""
        
        try:
            # 저장 경로
            save_path = os.path.join(self.cache_dir, f'{code}_{suffix}.csv')
            
            # 기존 파일 확인
            if os.path.exists(save_path) and suffix == 'stock_data':
                # 기존 데이터 로드
                existing_df = pd.read_csv(save_path)
                
                # 날짜 형식 변환
                if 'date' in existing_df.columns and 'date' in df.columns:
                    existing_df['date'] = pd.to_datetime(existing_df['date'])
                    df['date'] = pd.to_datetime(df['date'])
                    
                    # 중복 제거 및 병합
                    combined_df = pd.concat([existing_df, df])
                    combined_df = combined_df.drop_duplicates(subset=['date']).sort_values('date')
                    
                    # 저장
                    combined_df.to_csv(save_path, index=False)
                    logger.info(f"{code} 기존 데이터와 통합하여 저장 완료: {save_path}")
                    return save_path
            
            # 새로 저장
            df.to_csv(save_path, index=False)
            logger.info(f"{code} 데이터 저장 완료: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"데이터 저장 실패: {e}", exc_info=True)
            return ""
    
    def save_to_json(self, data: Dict[str, Any], code: str, suffix: str = 'data') -> str:
        """
        딕셔너리를 JSON 파일로 저장
        
        Args:
            data (dict): 저장할 데이터
            code (str): 종목 코드
            suffix (str): 파일명 접미사
            
        Returns:
            str: 저장된 파일 경로
        """
        if not data:
            logger.warning(f"{code} 저장할 데이터가 없습니다")
            return ""
        
        try:
            # 저장 경로
            save_path = os.path.join(self.cache_dir, f'{code}_{suffix}.json')
            
            # JSON으로 저장
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"{code} JSON 데이터 저장 완료: {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"JSON 데이터 저장 실패: {e}", exc_info=True)
            return ""

    # ----- 3단계: 외부 데이터 수집 함수들 -----
    
    def get_volume_rank_data(self) -> List[Dict[str, Any]]:
        """
        거래량 상위 종목 데이터 가져오기
        
        Returns:
            list: 거래량 상위 종목 목록
        """
        try:
            # 캐시 파일 경로
            cache_file = os.path.join(self.cache_dir, 'volume_rank_data.json')
            
            if os.path.exists(cache_file):
                # 캐시 유효성 확인 (8시간 이내)
                cache_time = datetime.datetime.fromtimestamp(os.path.getmtime(cache_file))
                current_time = datetime.datetime.now()
                
                if (current_time - cache_time).total_seconds() < 28800:  # 8시간
                    # 캐시 파일 읽기
                    with open(cache_file, 'r') as f:
                        volume_rank_data = json.load(f)
                    
                    logger.info(f"거래량 상위 데이터 캐시 사용 ({len(volume_rank_data)}개 종목)")
                    return volume_rank_data
                else:
                    logger.warning("거래량 상위 데이터 캐시가 오래됨")
            
            # 캐시가 없거나 오래된 경우 Excel 파일에서 데이터 가져오기
            data = self.fetch_volume_rank_from_excel()
            
            if data:
                # 캐시 파일 저장
                with open(cache_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"거래량 상위 데이터 업데이트 완료 ({len(data)}개 종목)")
                return data
            else:
                logger.warning("거래량 상위 데이터를 가져올 수 없습니다")
                return []
                
        except Exception as e:
            logger.error(f"거래량 상위 데이터 가져오기 실패: {e}", exc_info=True)
            return []
            
    # 기타 외부 데이터 수집 함수들은 원래 코드 유지...
    
    # ----- 4단계: 데이터 변환 및 통합 함수들 -----
    
    def prepare_data_for_rl(self, code: str) -> Dict[str, Any]:
        """
        강화학습을 위한 데이터 준비 (과거 데이터 + 실시간 데이터)
        
        Args:
            code (str): 종목 코드
            
        Returns:
            dict: 강화학습용 데이터
        """
        try:
            # 1. 주가 데이터 업데이트
            stock_data = self.update_stock_data(code)
            
            # 2. 최신 시장 데이터 가져오기
            market_data = self.get_latest_market_data(code)
            
            # 3. 거래량 상위 데이터 가져오기
            volume_rank_data = self.get_volume_rank_data()
            
            # 4. 결과 데이터 생성
            result = {
                'stock_data': stock_data.to_dict('records') if not stock_data.empty else [],
                'market_data': market_data,
                'volume_rank': volume_rank_data,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 5. 통합 데이터 캐시 저장
            self.save_to_json(result, code, suffix='rl_data')
            
            logger.info(f"{code} 강화학습 데이터 준비 완료")
            return result
            
        except Exception as e:
            logger.error(f"강화학습 데이터 준비 실패: {e}", exc_info=True)
            return {}