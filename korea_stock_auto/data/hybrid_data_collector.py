"""
한국 주식 자동매매 - 통합 데이터 수집기

한국투자증권 API와 네이버 증권 크롤러를 통합하여 
부족한 데이터를 서로 보완하는 기능을 제공합니다.
"""

import pandas as pd
import logging
import time
import requests
import re
import os
import pickle
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Any, Union
from bs4 import BeautifulSoup

from korea_stock_auto.api.client import KoreaInvestmentApiClient
from korea_stock_auto.data.database.price_data import PriceDataManager

logger = logging.getLogger(__name__)

class HybridDataCollector:
    """
    API와 크롤러를 통합한 하이브리드 데이터 수집기
    
    - 한국투자증권 API: 실시간 데이터, 정확한 최신 데이터
    - 네이버 증권 크롤러: 과거 장기 데이터, API 한계 보완
    - 시간 기반 캐싱: 장중/장 마감 후 다른 캐싱 전략 적용
    """
    
    def __init__(self, api_client: KoreaInvestmentApiClient = None, use_database: bool = True, api_delay: float = 0.1, crawler_delay: float = 0.5):
        """
        하이브리드 데이터 수집기 초기화
        
        Args:
            api_client: 기존 API 클라이언트 인스턴스 (없으면 새로 생성)
            use_database: 데이터베이스 사용 여부
            api_delay: API 호출 간 지연시간 (초)
            crawler_delay: 크롤러 요청 간 지연시간 (초)
        """
        if api_client is not None:
            self.api_client = api_client
        else:
            self.api_client = KoreaInvestmentApiClient()
        
        if use_database:
            self.db_manager = PriceDataManager()
        else:
            self.db_manager = None
        
        self.api_delay = api_delay
        self.crawler_delay = crawler_delay
        
        # 파일 기반 캐싱 시스템
        self.cache_dir = "cache/stock_data"
        self.api_cache_dir = "cache/api_data"  # API 전용 캐시 디렉토리
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.api_cache_dir, exist_ok=True)
        
        # 한국 주식시장 운영시간 설정
        self.market_open_time = dt_time(9, 0)    # 09:00
        self.market_close_time = dt_time(15, 30)  # 15:30
        
        # 네이버 크롤링용 세션 설정
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://finance.naver.com/',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        logger.info("하이브리드 데이터 수집기 초기화 완료 (시간 기반 캐싱)")
    
    def get_stock_data(self, code: str, period: str = "day", count: int = 1250, 
                      strategy: str = "auto") -> Optional[pd.DataFrame]:
        """
        종목 데이터 수집 (하이브리드 전략)
        
        Args:
            code: 종목 코드 (6자리)
            period: 조회 기간 ("day", "week", "month")
            count: 조회할 데이터 개수 (기본값: 1250개=약5년)
            strategy: 수집 전략
                - "auto": 데이터베이스 활성화시 db_first, 비활성화시 hybrid (기본값)
                - "hybrid": API 우선, 부족시 크롤러 보완
                - "api_only": API만 사용
                - "crawler_only": 크롤러만 사용
                - "db_first": 데이터베이스 우선, 부족시 API/크롤러
                
        Returns:
            pd.DataFrame: OHLCV 데이터프레임 또는 None
        """
        # auto 전략인 경우 데이터베이스 상태에 따라 자동 선택
        if strategy == "auto":
            if self.db_manager is not None:
                strategy = "db_first"
                logger.info(f"종목 데이터 수집 시작: {code}, {period}, {count}개, 전략: auto -> db_first")
            else:
                strategy = "hybrid"
                logger.info(f"종목 데이터 수집 시작: {code}, {period}, {count}개, 전략: auto -> hybrid")
        else:
            logger.info(f"종목 데이터 수집 시작: {code}, {period}, {count}개, 전략: {strategy}")
        
        # 캐시 확인
        cached_data = self._get_cached_data(code, period, count)
        if cached_data is not None:
            logger.info(f"캐시된 데이터 사용: {code}, {len(cached_data)}개 레코드")
            return cached_data
        
        # 캐시에 없으면 전략에 따라 데이터 수집
        result = None
        if strategy == "api_only":
            result = self._get_data_from_api(code, period, count)
        elif strategy == "crawler_only":
            result = self._get_data_from_crawler(code, period, count)
        elif strategy == "db_first":
            result = self._get_data_db_first(code, period, count)
        else:  # hybrid
            result = self._get_data_hybrid(code, period, count)
        
        # 결과를 캐시에 저장
        if result is not None:
            self._cache_data(code, period, count, result)
        
        return result
    
    def _get_data_from_api(self, code: str, period: str, count: int) -> Optional[pd.DataFrame]:
        """
        API에서 데이터 수집 (시간 기반 캐싱 적용)
        - 장 마감 후: 캐시된 API 데이터 우선 사용
        - 장중: 실시간 API 데이터 사용
        """
        try:
            # 1. 장 마감 후 또는 캐시 조건 확인
            if self.should_use_cached_api_data(code, period):
                cached_api_data = self._get_api_cache_data(code, period)
                if cached_api_data is not None and len(cached_api_data) > 0:
                    logger.info(f"API 캐시 데이터 사용: {code}, {len(cached_api_data)}개")
                    return cached_api_data.head(count) if len(cached_api_data) >= count else cached_api_data
            
            # 2. API 제한 확인
            if count > 30:
                logger.warning(f"API는 최대 30개까지만 지원합니다. 30개로 제한합니다.")
                count = 30
            
            # 3. 실시간 API 데이터 수집
            period_map = {"day": "D", "week": "W", "month": "M"}
            api_period = period_map.get(period, "D")
            
            logger.info(f"실시간 API 데이터 수집: {code}, {count}개")
            api_data = self.api_client.get_daily_stock_chart_data(
                symbol=code,
                period_div_code=api_period,
                limit=count
            )
            
            if api_data:
                df = self._convert_api_to_dataframe(api_data, code)
                if df is not None and not df.empty:
                    logger.info(f"API에서 {len(df)}개 데이터 수집 성공: {code}")
                    
                    # 4. 장 마감 후에는 API 데이터도 캐싱
                    self._cache_api_data(code, period, df)
                    
                    return df
                else:
                    logger.warning(f"API 데이터 변환 실패: {code}")
                    return None
            else:
                logger.warning(f"API에서 데이터 수집 실패: {code}")
                return None
                
        except Exception as e:
            logger.error(f"API 데이터 수집 오류: {e}", exc_info=True)
            return None
    
    def _get_data_from_crawler(self, code: str, period: str, count: int) -> Optional[pd.DataFrame]:
        """
        크롤러에서만 데이터 수집 (네이버 증권)
        """
        try:
            # 5년 제한 적용 (일봉 기준 약 1250개)
            max_days_5years = 1250
            if count == -1 or count > max_days_5years:
                count = max_days_5years
                logger.info(f"데이터 요청량을 5년치로 제한: {count}개")
            
            # 주봉/월봉의 경우 일봉 데이터를 더 많이 크롤링해서 변환
            if period == "week":
                daily_count = min(count * 7, max_days_5years)
            elif period == "month":
                daily_count = min(count * 30, max_days_5years)
            else:
                daily_count = count
            
            # 일봉 데이터 크롤링
            daily_data = self._crawl_naver_daily_data(code, daily_count)
            
            if daily_data is None or len(daily_data) == 0:
                logger.warning(f"일봉 데이터 크롤링 실패: {code}")
                return None
            
            # 기간에 따라 데이터 변환
            if period == "day":
                result = daily_data.head(count) if count > 0 else daily_data
            elif period == "week":
                weekly_data = self._convert_to_weekly(daily_data)
                result = weekly_data.head(count) if count > 0 else weekly_data
            elif period == "month":
                monthly_data = self._convert_to_monthly(daily_data)
                result = monthly_data.head(count) if count > 0 else monthly_data
            else:
                logger.error(f"지원하지 않는 기간: {period}")
                return None
            
            logger.info(f"네이버 증권 {period} 데이터 처리 완료: {code}, {len(result)}개 레코드")
            return result
                
        except Exception as e:
            logger.error(f"네이버 증권 데이터 처리 오류: {e}", exc_info=True)
            return None
    
    def _crawl_naver_daily_data(self, code: str, count: int) -> Optional[pd.DataFrame]:
        """
        네이버 증권에서 일봉 데이터 크롤링
        
        Args:
            code: 종목 코드 (6자리)
            count: 조회할 데이터 개수 (최대 5년치=1250개)
            
        Returns:
            pd.DataFrame: 일봉 OHLCV 데이터프레임 또는 None
        """
        try:
            base_url = f"https://finance.naver.com/item/sise_day.naver?code={code}"
            
            max_days_5years = 1250
            if count > max_days_5years:
                count = max_days_5years
                logger.info(f"크롤링 데이터를 5년치로 제한: {count}개")
            
            logger.info(f"네이버 증권 일별 시세 크롤링 시작: {code} ({count}개, 최대 5년)")
            
            all_data = []
            page = 1
            max_pages = min(150, (count // 10) + 20)
            target_count = count
            consecutive_empty_pages = 0
            max_empty_pages = 3
            
            five_years_ago = datetime.now() - timedelta(days=5*365)
            
            while len(all_data) < target_count and page <= max_pages and consecutive_empty_pages < max_empty_pages:
                page_url = f"{base_url}&page={page}"
                
                logger.debug(f"페이지 {page} 크롤링: {page_url}")
                
                try:
                    response = self.session.get(page_url, timeout=15)
                    response.raise_for_status()
                    
                    if response.status_code == 200:
                        page_data = self._parse_naver_html(response.text, code)
                        if page_data is not None and len(page_data) > 0:
                            filtered_data = []
                            for _, row in page_data.iterrows():
                                try:
                                    row_date = datetime.strptime(row['date'], '%Y-%m-%d')
                                    if row_date >= five_years_ago:
                                        filtered_data.append(row.to_dict())
                                    else:
                                        logger.info(f"5년 전 데이터 도달로 크롤링 중단: {row['date']}")
                                        consecutive_empty_pages = max_empty_pages
                                        break
                                except ValueError:
                                    continue
                            
                            if filtered_data:
                                all_data.extend(filtered_data)
                                logger.debug(f"페이지 {page}에서 {len(filtered_data)}개 데이터 수집 (총 {len(all_data)}개)")
                                consecutive_empty_pages = 0
                                
                                if page % 10 == 0:
                                    logger.info(f"크롤링 진행: {page}페이지, {len(all_data)}개 데이터 수집")
                            else:
                                consecutive_empty_pages += 1
                                logger.debug(f"페이지 {page}에서 유효 데이터 없음 (연속 빈 페이지: {consecutive_empty_pages})")
                        else:
                            consecutive_empty_pages += 1
                            logger.debug(f"페이지 {page}에서 데이터 없음 (연속 빈 페이지: {consecutive_empty_pages})")
                    else:
                        logger.error(f"네이버 증권 HTTP 오류: {response.status_code}")
                        consecutive_empty_pages += 1
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"페이지 {page} 요청 오류: {e}")
                    consecutive_empty_pages += 1
                
                page += 1
                time.sleep(self.crawler_delay)
            
            if consecutive_empty_pages >= max_empty_pages:
                logger.info(f"연속 빈 페이지 {max_empty_pages}개 감지 또는 5년 전 데이터 도달로 크롤링 종료")
            elif len(all_data) >= target_count:
                logger.info(f"목표 데이터 수 달성으로 크롤링 종료")
            elif page > max_pages:
                logger.info(f"최대 페이지 수 도달로 크롤링 종료")
            
            if all_data:
                if count > 0:
                    df = pd.DataFrame(all_data[:count])
                else:
                    df = pd.DataFrame(all_data)
                    
                df = df.drop_duplicates(subset=['date'])
                df = df.sort_values('date', ascending=False).reset_index(drop=True)
                
                logger.info(f"네이버 증권 일봉 데이터 크롤링 성공: {code}, {len(df)}개 레코드")
                return df
            else:
                logger.warning(f"네이버 증권 일봉 데이터 크롤링 실패: {code}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"네이버 증권 요청 오류: {e}")
            return None
        except Exception as e:
            logger.error(f"네이버 증권 크롤링 오류: {e}", exc_info=True)
            return None
    
    def _parse_naver_html(self, html_content: str, code: str) -> Optional[pd.DataFrame]:
        """
        네이버 증권 HTML 페이지 파싱
        
        Args:
            html_content: HTML 내용
            code: 종목 코드
            
        Returns:
            pd.DataFrame: 파싱된 OHLCV 데이터 또는 None
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            table = soup.find('table', {'class': 'type2'})
            if not table:
                logger.warning(f"네이버 증권 시세 테이블을 찾을 수 없음: {code}")
                return None
            
            rows = table.find('tbody').find_all('tr') if table.find('tbody') else table.find_all('tr')[1:]
            
            data_list = []
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 7:
                    try:
                        date_text = cols[0].get_text(strip=True)
                        close_text = cols[1].get_text(strip=True)
                        change_text = cols[2].get_text(strip=True)
                        open_text = cols[3].get_text(strip=True)
                        high_text = cols[4].get_text(strip=True)
                        low_text = cols[5].get_text(strip=True)
                        volume_text = cols[6].get_text(strip=True)
                        
                        if not date_text or date_text.isspace():
                            continue
                        
                        close_price = float(close_text.replace(',', '')) if close_text and close_text != '' else 0
                        open_price = float(open_text.replace(',', '')) if open_text and open_text != '' else 0
                        high_price = float(high_text.replace(',', '')) if high_text and high_text != '' else 0
                        low_price = float(low_text.replace(',', '')) if low_text and low_text != '' else 0
                        volume = int(volume_text.replace(',', '')) if volume_text and volume_text != '' else 0
                        
                        if '.' in date_text:
                            formatted_date = date_text.replace('.', '-')
                        else:
                            formatted_date = date_text
                        
                        if close_price > 0:
                            data_list.append({
                                'date': formatted_date,
                                'open': open_price,
                                'high': high_price,
                                'low': low_price,
                                'close': close_price,
                                'volume': volume
                            })
                        
                    except (ValueError, IndexError, AttributeError) as e:
                        logger.warning(f"네이버 증권 행 파싱 오류: {e}")
                        continue
            
            if not data_list:
                logger.warning(f"네이버 증권 파싱된 데이터 없음: {code}")
                return None
            
            df = pd.DataFrame(data_list)
            logger.debug(f"네이버 증권 HTML 파싱 완료: {code}, {len(df)}개 레코드")
            return df
            
        except Exception as e:
            logger.error(f"네이버 증권 HTML 파싱 오류: {e}", exc_info=True)
            return None
    
    def _convert_api_to_dataframe(self, api_data: Any, symbol: str) -> Optional[pd.DataFrame]:
        """
        API 응답을 DataFrame으로 변환
        """
        try:
            # 디버깅: API 응답 구조 확인
            logger.debug(f"API 응답 타입: {type(api_data)}")
            logger.debug(f"API 응답 내용: {api_data}")
            
            # API 응답이 리스트인 경우 (get_daily_stock_chart_data의 경우)
            if isinstance(api_data, list):
                if not api_data:
                    logger.warning(f"API 응답 리스트가 비어있음: {symbol}")
                    return None
                
                output = api_data
                logger.debug(f"리스트 형태 API 응답 처리: {len(output)}개 항목")
                
            # API 응답이 딕셔너리인 경우 (기존 방식)
            elif isinstance(api_data, dict):
                # output 또는 output2 키 확인
                output = api_data.get('output', [])
                if not output:
                    output = api_data.get('output2', [])
                if not output:
                    output = api_data.get('output1', [])
                
                if not output:
                    logger.warning(f"API 응답에 output 데이터가 없음: {symbol}")
                    logger.warning(f"사용 가능한 키: {list(api_data.keys())}")
                    return None
                    
                logger.debug(f"딕셔너리 형태 API 응답 처리: {len(output)}개 항목")
            else:
                logger.warning(f"API 응답이 비어있거나 형식이 잘못됨: {symbol}")
                logger.warning(f"응답 타입: {type(api_data)}, 내용: {api_data}")
                return None
            
            logger.debug(f"Output 데이터 타입: {type(output)}, 길이: {len(output) if isinstance(output, list) else 'N/A'}")
            if isinstance(output, list) and len(output) > 0:
                logger.debug(f"첫 번째 항목 키: {list(output[0].keys()) if isinstance(output[0], dict) else 'N/A'}")
                logger.debug(f"첫 번째 항목: {output[0]}")
            
            data_list = []
            for i, item in enumerate(output):
                try:
                    if not isinstance(item, dict):
                        logger.warning(f"항목 {i}이 딕셔너리가 아님: {type(item)}")
                        continue
                    
                    date_str = item.get('stck_bsop_date', '')
                    if len(date_str) == 8:
                        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    else:
                        formatted_date = date_str
                    
                    # 가격 데이터 추출 시 문자열 처리
                    open_price = item.get('stck_oprc', '0')
                    high_price = item.get('stck_hgpr', '0')
                    low_price = item.get('stck_lwpr', '0')
                    close_price = item.get('stck_clpr', '0')
                    volume = item.get('acml_vol', '0')
                    
                    # 문자열에서 숫자 추출 (콤마 제거)
                    if isinstance(open_price, str):
                        open_price = open_price.replace(',', '')
                    if isinstance(high_price, str):
                        high_price = high_price.replace(',', '')
                    if isinstance(low_price, str):
                        low_price = low_price.replace(',', '')
                    if isinstance(close_price, str):
                        close_price = close_price.replace(',', '')
                    if isinstance(volume, str):
                        volume = volume.replace(',', '')
                    
                    data_list.append({
                        'date': formatted_date,
                        'open': float(open_price) if open_price else 0.0,
                        'high': float(high_price) if high_price else 0.0,
                        'low': float(low_price) if low_price else 0.0,
                        'close': float(close_price) if close_price else 0.0,
                        'volume': int(float(volume)) if volume else 0
                    })
                except (ValueError, KeyError) as e:
                    logger.warning(f"API 데이터 항목 {i} 변환 오류: {e}")
                    logger.warning(f"문제 항목: {item}")
                    continue
            
            if not data_list:
                logger.warning(f"변환 가능한 API 데이터가 없음: {symbol}")
                return None
            
            df = pd.DataFrame(data_list)
            df = df.sort_values('date', ascending=False).reset_index(drop=True)
            
            logger.info(f"API 데이터 변환 완료: {symbol}, {len(df)}개")
            return df
            
        except Exception as e:
            logger.error(f"API 데이터 변환 오류: {e}", exc_info=True)
            return None
    
    def _convert_to_weekly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        일봉 데이터를 주봉 데이터로 변환
        
        Args:
            daily_df: 일봉 데이터프레임
            
        Returns:
            pd.DataFrame: 주봉 데이터프레임
        """
        try:
            if daily_df.empty:
                return daily_df
            
            df = daily_df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.sort_index()
            
            weekly = df.resample('W-MON').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            weekly = weekly.reset_index()
            weekly['date'] = weekly['date'].dt.strftime('%Y-%m-%d')
            
            weekly = weekly.sort_values('date', ascending=False).reset_index(drop=True)
            
            logger.debug(f"일봉 → 주봉 변환 완료: {len(daily_df)} → {len(weekly)}")
            return weekly
            
        except Exception as e:
            logger.error(f"주봉 변환 오류: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _convert_to_monthly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        일봉 데이터를 월봉 데이터로 변환
        
        Args:
            daily_df: 일봉 데이터프레임
            
        Returns:
            pd.DataFrame: 월봉 데이터프레임
        """
        try:
            if daily_df.empty:
                return daily_df
            
            df = daily_df.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.sort_index()
            
            monthly = df.resample('M').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            monthly = monthly.reset_index()
            monthly['date'] = monthly['date'].dt.strftime('%Y-%m-%d')
            
            monthly = monthly.sort_values('date', ascending=False).reset_index(drop=True)
            
            logger.debug(f"일봉 → 월봉 변환 완료: {len(daily_df)} → {len(monthly)}")
            return monthly
            
        except Exception as e:
            logger.error(f"월봉 변환 오류: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _get_data_hybrid(self, code: str, period: str, count: int) -> Optional[pd.DataFrame]:
        """
        하이브리드 전략: API 우선, 부족시 크롤러 보완
        """
        try:
            api_count = min(count, 30)
            api_data = self._get_data_from_api(code, period, api_count)
            
            if count <= 30 and api_data is not None:
                return api_data
            
            logger.info(f"API 데이터 부족 또는 실패, 크롤러로 {count}개 데이터 수집 시도")
            crawler_data = self._get_data_from_crawler(code, period, count)
            
            if crawler_data is not None:
                if api_data is not None:
                    combined_data = self._merge_api_crawler_data(api_data, crawler_data)
                    logger.info(f"API + 크롤러 데이터 병합 완료: {len(combined_data)}개")
                    return combined_data
                else:
                    return crawler_data
            
            logger.error(f"모든 데이터 수집 방법 실패: {code}")
            return None
            
        except Exception as e:
            logger.error(f"하이브리드 데이터 수집 오류: {e}", exc_info=True)
            return None
    
    def _get_data_db_first(self, code: str, period: str, count: int) -> Optional[pd.DataFrame]:
        """
        데이터베이스 우선 전략 - 오늘 기준 필요한 날짜 범위 스마트 분석
        """
        if not self.db_manager:
            logger.warning("데이터베이스가 비활성화되어 있습니다. 하이브리드 전략으로 대체합니다.")
            return self._get_data_hybrid(code, period, count)
        
        try:
            # 1. 오늘 기준 필요한 데이터 범위 분석
            target_date_range = self._calculate_target_date_range(count, period)
            logger.info(f"필요한 데이터 범위: {target_date_range['start_date']} ~ {target_date_range['end_date']}")
            
            # 2. DB에서 해당 범위의 데이터 확인
            db_coverage = self._analyze_db_coverage(code, target_date_range, count)
            
            if db_coverage['is_sufficient']:
                logger.info(f"DB에서 충분한 데이터 발견: {db_coverage['available_count']}개")
                return db_coverage['data'].head(count)
            
            # 3. 부족한 데이터 범위 계산 및 보완 전략 결정
            missing_info = self._calculate_missing_data(db_coverage, target_date_range)
            logger.info(f"부족한 데이터: {missing_info['missing_days']}일 ({missing_info['missing_recent']}일 최신 + {missing_info['missing_old']}일 과거)")
            
            # 4. 최신 데이터 보완 (API 우선)
            recent_data = None
            if missing_info['missing_recent'] > 0:
                api_count = min(30, missing_info['missing_recent'])
                recent_data = self._get_data_from_api(code, period, api_count)
                logger.info(f"API로 최신 {api_count}일 데이터 수집")
            
            # 5. 과거 데이터 보완 (크롤링 필요시에만)
            old_data = None
            if missing_info['missing_old'] > 0:
                # API로 충분하지 않은 경우에만 크롤링
                total_needed = missing_info['missing_recent'] + missing_info['missing_old']
                if total_needed > 30:  # API 한계 초과시
                    logger.info(f"API 한계 초과, 크롤링으로 {total_needed}일 데이터 수집")
                    old_data = self._get_data_from_crawler(code, period, total_needed)
                else:
                    logger.info(f"API만으로 충분, 크롤링 생략")
            
            # 6. 데이터 병합 및 저장
            final_data = self._merge_all_data(recent_data, old_data, db_coverage['data'])
            
            if final_data is not None and not final_data.empty:
                # 새로운 데이터만 DB에 저장
                self._save_new_data_to_db(code, final_data, db_coverage['data'])
                
                return final_data.head(count) if len(final_data) >= count else final_data
            
            # 7. 실패시 기존 DB 데이터라도 반환
            if not db_coverage['data'].empty:
                logger.warning(f"데이터 수집 실패, 기존 DB 데이터 반환: {len(db_coverage['data'])}개")
                return db_coverage['data']
            
            return None
            
        except Exception as e:
            logger.error(f"데이터베이스 우선 수집 오류: {e}", exc_info=True)
            return self._get_data_hybrid(code, period, count)

    def _merge_api_crawler_data(self, api_data: pd.DataFrame, crawler_data: pd.DataFrame) -> pd.DataFrame:
        """
        API 데이터와 크롤러 데이터 병합 (중복 제거, 최신 순 정렬)
        """
        try:
            combined = pd.concat([api_data, crawler_data], ignore_index=True)
            
            combined = combined.drop_duplicates(subset=['date'], keep='first')
            
            combined = combined.sort_values('date', ascending=False).reset_index(drop=True)
            
            logger.debug(f"데이터 병합 완료: API {len(api_data)}개 + 크롤러 {len(crawler_data)}개 -> {len(combined)}개")
            return combined
            
        except Exception as e:
            logger.error(f"데이터 병합 오류: {e}", exc_info=True)
            return crawler_data if not crawler_data.empty else api_data
    
    def _merge_api_db_data(self, api_data: pd.DataFrame, db_data: pd.DataFrame) -> pd.DataFrame:
        """
        API 데이터와 DB 데이터 병합 (중복 제거, 최신 순 정렬)
        """
        try:
            combined = pd.concat([api_data, db_data], ignore_index=True)
            
            combined = combined.drop_duplicates(subset=['date'], keep='first')
            
            combined = combined.sort_values('date', ascending=False).reset_index(drop=True)
            
            logger.debug(f"API-DB 데이터 병합 완료: API {len(api_data)}개 + DB {len(db_data)}개 -> {len(combined)}개")
            return combined
            
        except Exception as e:
            logger.error(f"API-DB 데이터 병합 오류: {e}", exc_info=True)
            return db_data if not db_data.empty else api_data
    
    def _filter_new_data(self, api_data: pd.DataFrame, db_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        API 데이터 중 DB에 없는 새로운 데이터만 필터링
        """
        try:
            if api_data.empty:
                return None
            
            if db_data.empty:
                return api_data
            
            # DB에 있는 날짜 목록
            existing_dates = set(db_data['date'].tolist())
            
            # API 데이터 중 DB에 없는 날짜만 필터링
            new_data = api_data[~api_data['date'].isin(existing_dates)]
            
            if not new_data.empty:
                logger.debug(f"새로운 데이터 필터링 완료: API {len(api_data)}개 -> 새로운 데이터 {len(new_data)}개")
                return new_data
            else:
                logger.debug("새로운 데이터 없음")
                return None
                
        except Exception as e:
            logger.error(f"새로운 데이터 필터링 오류: {e}", exc_info=True)
            return api_data
    
    def _calculate_target_date_range(self, count: int, period: str) -> Dict[str, Any]:
        """
        오늘 기준으로 필요한 데이터 날짜 범위 계산
        """
        from datetime import datetime, timedelta
        
        try:
            end_date = datetime.now().date()
            
            # 거래일 기준으로 계산 (주말 제외, 공휴일은 단순화)
            if period == "day":
                # 일봉: 약 1.4배 여유 (주말 고려)
                calendar_days = int(count * 1.4)
            elif period == "week":
                # 주봉: 7배
                calendar_days = count * 7
            elif period == "month":
                # 월봉: 30배
                calendar_days = count * 30
            else:
                calendar_days = int(count * 1.4)
            
            start_date = end_date - timedelta(days=calendar_days)
            
            return {
                'start_date': start_date,
                'end_date': end_date,
                'target_count': count,
                'period': period
            }
            
        except Exception as e:
            logger.error(f"날짜 범위 계산 오류: {e}", exc_info=True)
            # 기본값 반환
            return {
                'start_date': datetime.now().date() - timedelta(days=count * 2),
                'end_date': datetime.now().date(),
                'target_count': count,
                'period': period
            }
    
    def _analyze_db_coverage(self, code: str, target_range: Dict[str, Any], count: int) -> Dict[str, Any]:
        """
        DB에서 목표 날짜 범위의 데이터 커버리지 분석
        """
        try:
            # DB에서 넉넉하게 데이터 조회
            db_data = self.db_manager.get_price_history(code, count * 2)
            
            if db_data.empty:
                return {
                    'is_sufficient': False,
                    'data': pd.DataFrame(),
                    'available_count': 0,
                    'coverage_start': None,
                    'coverage_end': None,
                    'missing_recent_days': count,
                    'missing_old_days': 0
                }
            
            # 날짜 컬럼을 datetime으로 변환
            db_data['date'] = pd.to_datetime(db_data['date']).dt.date
            db_data = db_data.sort_values('date', ascending=False)
            
            # 목표 범위 내 데이터만 필터링
            target_data = db_data[
                (db_data['date'] >= target_range['start_date']) & 
                (db_data['date'] <= target_range['end_date'])
            ]
            
            is_sufficient = len(target_data) >= count
            
            return {
                'is_sufficient': is_sufficient,
                'data': target_data,
                'available_count': len(target_data),
                'coverage_start': target_data['date'].min() if not target_data.empty else None,
                'coverage_end': target_data['date'].max() if not target_data.empty else None,
                'all_db_data': db_data  # 전체 DB 데이터도 저장
            }
            
        except Exception as e:
            logger.error(f"DB 커버리지 분석 오류: {e}", exc_info=True)
            return {
                'is_sufficient': False,
                'data': pd.DataFrame(),
                'available_count': 0,
                'coverage_start': None,
                'coverage_end': None
            }
    
    def _calculate_missing_data(self, db_coverage: Dict[str, Any], target_range: Dict[str, Any]) -> Dict[str, Any]:
        """
        부족한 데이터 범위 계산
        """
        try:
            from datetime import datetime, timedelta
            
            target_count = target_range['target_count']
            available_count = db_coverage['available_count']
            
            if available_count >= target_count:
                return {
                    'missing_days': 0,
                    'missing_recent': 0,
                    'missing_old': 0
                }
            
            missing_days = target_count - available_count
            
            # 최신 데이터 부족분 계산
            today = datetime.now().date()
            if db_coverage['coverage_end']:
                days_since_last = (today - db_coverage['coverage_end']).days
                missing_recent = min(missing_days, max(0, days_since_last))
            else:
                missing_recent = min(missing_days, 30)  # API 한계
            
            # 과거 데이터 부족분
            missing_old = max(0, missing_days - missing_recent)
            
            return {
                'missing_days': missing_days,
                'missing_recent': missing_recent,
                'missing_old': missing_old
            }
            
        except Exception as e:
            logger.error(f"부족한 데이터 계산 오류: {e}", exc_info=True)
            return {
                'missing_days': target_range['target_count'],
                'missing_recent': 30,
                'missing_old': max(0, target_range['target_count'] - 30)
            }
    
    def _merge_all_data(self, recent_data: Optional[pd.DataFrame], old_data: Optional[pd.DataFrame], db_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        최신 데이터, 과거 데이터, DB 데이터를 모두 병합
        """
        try:
            data_pieces = []
            
            if recent_data is not None and not recent_data.empty:
                data_pieces.append(recent_data)
                logger.debug(f"최신 데이터 추가: {len(recent_data)}개")
            
            if old_data is not None and not old_data.empty:
                data_pieces.append(old_data)
                logger.debug(f"과거 데이터 추가: {len(old_data)}개")
            
            if not db_data.empty:
                data_pieces.append(db_data)
                logger.debug(f"DB 데이터 추가: {len(db_data)}개")
            
            if not data_pieces:
                return None
            
            # 모든 데이터 병합
            combined = pd.concat(data_pieces, ignore_index=True)
            
            # 중복 제거 및 정렬
            combined = combined.drop_duplicates(subset=['date'], keep='first')
            combined = combined.sort_values('date', ascending=False).reset_index(drop=True)
            
            logger.info(f"전체 데이터 병합 완료: {len(combined)}개")
            return combined
            
        except Exception as e:
            logger.error(f"전체 데이터 병합 오류: {e}", exc_info=True)
            return db_data if not db_data.empty else None
    
    def _save_new_data_to_db(self, code: str, final_data: pd.DataFrame, existing_db_data: pd.DataFrame) -> bool:
        """
        새로운 데이터만 DB에 저장
        """
        try:
            if existing_db_data.empty:
                # DB에 데이터가 없으면 전체 저장
                return self._save_to_database(code, final_data)
            
            # 기존 DB 날짜 목록
            existing_dates = set()
            if 'date' in existing_db_data.columns:
                existing_dates = set(pd.to_datetime(existing_db_data['date']).dt.date.tolist())
            
            # 새로운 데이터만 필터링
            final_data['date_obj'] = pd.to_datetime(final_data['date']).dt.date
            new_data = final_data[~final_data['date_obj'].isin(existing_dates)]
            new_data = new_data.drop('date_obj', axis=1)
            
            if not new_data.empty:
                success = self._save_to_database(code, new_data)
                if success:
                    logger.info(f"새로운 데이터 {len(new_data)}개를 DB에 저장 완료")
                return success
            else:
                logger.debug("저장할 새로운 데이터 없음")
                return True
                
        except Exception as e:
            logger.error(f"새 데이터 DB 저장 오류: {e}", exc_info=True)
            return False
    
    def _save_to_database(self, code: str, data: pd.DataFrame) -> bool:
        """
        데이터를 데이터베이스에 저장
        """
        if not self.db_manager:
            return False
        
        try:
            success_count = 0
            for _, row in data.iterrows():
                price_data = {
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'volume': row['volume']
                }
                
                if self.db_manager.save_price_data(code, row['date'], price_data):
                    success_count += 1
            
            logger.info(f"데이터베이스 저장 완료: {code}, {success_count}/{len(data)}개")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"데이터베이스 저장 오류: {e}", exc_info=True)
            return False
    
    def get_multiple_stocks_data(self, codes: List[str], period: str = "day", 
                                count: int = 1250, strategy: str = "auto") -> Dict[str, pd.DataFrame]:
        """
        여러 종목의 데이터를 순차적으로 수집
        
        Args:
            codes: 종목 코드 리스트
            period: 조회 기간
            count: 조회할 데이터 개수 (기본값: 1250개=약5년)
            strategy: 수집 전략
            
        Returns:
            Dict[str, pd.DataFrame]: 종목별 데이터 딕셔너리
        """
        result = {}
        
        logger.info(f"다중 종목 데이터 수집 시작: {len(codes)}개 종목, 전략: {strategy}")
        
        for i, code in enumerate(codes, 1):
            logger.info(f"수집 진행: {i}/{len(codes)} - {code}")
            
            data = self.get_stock_data(code, period, count, strategy)
            if data is not None:
                result[code] = data
                logger.info(f"수집 성공: {code}, {len(data)}개 레코드")
            else:
                logger.error(f"수집 실패: {code}")
        
        logger.info(f"다중 종목 수집 완료: {len(result)}/{len(codes)}개 성공")
        return result
    
    def get_technical_analysis_data(self, code: str, period: str = "day", 
                                   days: int = 1250) -> Optional[pd.DataFrame]:
        """
        기술적 분석용 데이터 수집 (충분한 과거 데이터 확보)
        
        Args:
            code: 종목 코드
            period: 조회 기간
            days: 필요한 일수 (기본값: 1250일=약5년)
            
        Returns:
            pd.DataFrame: 기술적 분석용 OHLCV 데이터
        """
        logger.info(f"기술적 분석용 데이터 수집: {code}, {days}일")
        
        data = self.get_stock_data(code, period, days, strategy="auto")
        
        if data is not None and len(data) >= days:
            logger.info(f"기술적 분석 데이터 준비 완료: {code}, {len(data)}개")
            return data
        else:
            logger.warning(f"기술적 분석 데이터 부족: {code}, 요청 {days}개, 수집 {len(data) if data is not None else 0}개")
            return data
    
    def _get_cached_data(self, code: str, period: str, count: int) -> Optional[pd.DataFrame]:
        """
        캐시된 데이터 조회 (파일 기반)
        더 큰 count의 캐시 파일이 있으면 그것을 사용
        """
        try:
            # 정확한 캐시 파일 먼저 확인
            cache_file = f"{self.cache_dir}/{code}_{period}_{count}.pkl"
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.debug(f"정확한 캐시 파일 사용: {cache_file}")
                return cached_data.head(count) if len(cached_data) >= count else cached_data
            
            # 더 큰 count의 캐시 파일 찾기
            cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith(f"{code}_{period}_") and f.endswith('.pkl')]
            
            best_file = None
            best_count = 0
            
            for file in cache_files:
                try:
                    # 파일명에서 count 추출: code_period_count.pkl
                    file_count = int(file.split('_')[-1].replace('.pkl', ''))
                    if file_count >= count and file_count > best_count:
                        best_count = file_count
                        best_file = file
                except ValueError:
                    continue
            
            if best_file:
                cache_path = os.path.join(self.cache_dir, best_file)
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.debug(f"더 큰 캐시 파일 사용: {best_file} ({best_count}개 -> {count}개 요청)")
                return cached_data.head(count) if len(cached_data) >= count else cached_data
            
            return None
            
        except Exception as e:
            logger.error(f"캐시 조회 오류: {e}", exc_info=True)
            return None
    
    def _cache_data(self, code: str, period: str, count: int, dataframe: pd.DataFrame):
        """
        데이터를 파일에 캐시 저장
        더 작은 count의 기존 캐시 파일들은 삭제
        """
        try:
            # 새 캐시 파일 저장
            cache_file = f"{self.cache_dir}/{code}_{period}_{count}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(dataframe, f)
            
            # 더 작은 count의 기존 캐시 파일들 삭제
            cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith(f"{code}_{period}_") and f.endswith('.pkl')]
            
            for file in cache_files:
                try:
                    file_count = int(file.split('_')[-1].replace('.pkl', ''))
                    if file_count < count:
                        old_file_path = os.path.join(self.cache_dir, file)
                        os.remove(old_file_path)
                        logger.debug(f"작은 캐시 파일 삭제: {file}")
                except ValueError:
                    continue
            
            logger.debug(f"데이터 캐시 저장: {cache_file}, {len(dataframe)}개 레코드")
            
        except Exception as e:
            logger.error(f"데이터 캐시 저장 오류: {e}", exc_info=True)
    
    def clear_cache(self, code: str = None, period: str = None, include_api_cache: bool = True):
        """
        캐시 파일 삭제
        
        Args:
            code: 특정 종목 코드 (None이면 전체)
            period: 특정 기간 (None이면 전체)
            include_api_cache: API 캐시도 함께 삭제할지 여부
        """
        try:
            deleted_count = 0
            
            # 1. 크롤러 캐시 삭제
            if os.path.exists(self.cache_dir):
                if code and period:
                    pattern = f"{code}_{period}_"
                elif code:
                    pattern = f"{code}_"
                else:
                    pattern = ""
                
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.pkl') and file.startswith(pattern):
                        file_path = os.path.join(self.cache_dir, file)
                        os.remove(file_path)
                        deleted_count += 1
            
            # 2. API 캐시 삭제
            if include_api_cache and os.path.exists(self.api_cache_dir):
                if code and period:
                    pattern = f"{code}_{period}_"
                elif code:
                    pattern = f"{code}_"
                else:
                    pattern = ""
                
                for file in os.listdir(self.api_cache_dir):
                    if file.endswith('.pkl') and file.startswith(pattern):
                        file_path = os.path.join(self.api_cache_dir, file)
                        os.remove(file_path)
                        deleted_count += 1
            
            logger.info(f"캐시 파일 {deleted_count}개 삭제 완료")
            
        except Exception as e:
            logger.error(f"캐시 삭제 오류: {e}", exc_info=True)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        캐시 상태 정보 반환 (크롤러 캐시 + API 캐시)
        """
        try:
            cache_info = {
                "crawler_cache": self._get_cache_dir_info(self.cache_dir),
                "api_cache": self._get_cache_dir_info(self.api_cache_dir),
                "market_status": {
                    "is_market_open": self.is_market_open(),
                    "current_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "market_open": "09:00",
                    "market_close": "15:30"
                }
            }
            
            # 전체 통계 계산
            total_files = cache_info["crawler_cache"]["total_files"] + cache_info["api_cache"]["total_files"]
            total_size = cache_info["crawler_cache"]["total_size_mb"] + cache_info["api_cache"]["total_size_mb"]
            
            cache_info["total_summary"] = {
                "total_files": total_files,
                "total_size_mb": round(total_size, 2)
            }
            
            return cache_info
            
        except Exception as e:
            logger.error(f"캐시 정보 조회 오류: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _get_cache_dir_info(self, cache_dir: str) -> Dict[str, Any]:
        """
        특정 캐시 디렉토리의 정보 조회
        """
        try:
            if not os.path.exists(cache_dir):
                return {"total_files": 0, "total_size_mb": 0, "cache_dir": cache_dir, "stocks": {}}
            
            cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
            total_size = 0
            stocks_info = {}
            
            for file in cache_files:
                file_path = os.path.join(cache_dir, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                
                # 파일명 파싱
                try:
                    if cache_dir == self.api_cache_dir:
                        # API 캐시: code_period_YYYY-MM-DD.pkl
                        parts = file.replace('.pkl', '').split('_')
                        if len(parts) >= 3:
                            code = parts[0]
                            period = parts[1]
                            date_str = '_'.join(parts[2:])  # 날짜 부분
                            
                            if code not in stocks_info:
                                stocks_info[code] = {}
                            if period not in stocks_info[code]:
                                stocks_info[code][period] = []
                            
                            stocks_info[code][period].append({
                                "date": date_str,
                                "size_kb": round(file_size / 1024, 2),
                                "file": file,
                                "type": "api_cache"
                            })
                    else:
                        # 크롤러 캐시: code_period_count.pkl
                        parts = file.replace('.pkl', '').split('_')
                        if len(parts) >= 3:
                            code = parts[0]
                            period = parts[1]
                            count = int(parts[2])
                            
                            if code not in stocks_info:
                                stocks_info[code] = {}
                            if period not in stocks_info[code]:
                                stocks_info[code][period] = []
                            
                            stocks_info[code][period].append({
                                "count": count,
                                "size_kb": round(file_size / 1024, 2),
                                "file": file,
                                "type": "crawler_cache"
                            })
                except (ValueError, IndexError):
                    continue
            
            return {
                "total_files": len(cache_files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_dir": cache_dir,
                "stocks": stocks_info
            }
            
        except Exception as e:
            logger.error(f"캐시 디렉토리 정보 조회 오류: {e}", exc_info=True)
            return {"error": str(e)}

    def close(self):
        """
        리소스 정리
        """
        if hasattr(self, 'session'):
            self.session.close()
        
        if hasattr(self, 'db_manager') and self.db_manager:
            if hasattr(self.db_manager, 'close'):
                self.db_manager.close()
        
        logger.info("하이브리드 데이터 수집기 종료")

    def is_market_open(self) -> bool:
        """
        현재 시간이 장 운영시간인지 확인
        
        Returns:
            bool: 장 운영시간이면 True, 아니면 False
        """
        now = datetime.now()
        current_time = now.time()
        
        # 주말 체크 (토요일=5, 일요일=6)
        if now.weekday() >= 5:
            return False
        
        # 장 운영시간 체크 (09:00 ~ 15:30)
        return self.market_open_time <= current_time <= self.market_close_time
    
    def should_use_cached_api_data(self, code: str, period: str = "day") -> bool:
        """
        API 캐시 데이터를 사용할지 결정
        
        Args:
            code: 종목 코드
            period: 조회 기간
            
        Returns:
            bool: 캐시 사용 여부
        """
        # 장 마감 후에는 항상 캐시 사용 (당일 데이터 확정)
        if not self.is_market_open():
            return True
        
        # 장중에는 API 캐시가 오늘 날짜인지 확인
        today = datetime.now().strftime('%Y-%m-%d')
        api_cache_file = f"{self.api_cache_dir}/{code}_{period}_{today}.pkl"
        
        if os.path.exists(api_cache_file):
            # 캐시 파일이 오늘 15:30 이후에 생성되었는지 확인
            file_time = datetime.fromtimestamp(os.path.getmtime(api_cache_file))
            market_close_today = datetime.combine(datetime.now().date(), self.market_close_time)
            
            if file_time >= market_close_today:
                logger.debug(f"장 마감 후 생성된 API 캐시 사용: {code}")
                return True
        
        return False
    
    def _get_api_cache_data(self, code: str, period: str = "day") -> Optional[pd.DataFrame]:
        """
        API 캐시 데이터 조회
        
        Args:
            code: 종목 코드
            period: 조회 기간
            
        Returns:
            pd.DataFrame: 캐시된 API 데이터 또는 None
        """
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            api_cache_file = f"{self.api_cache_dir}/{code}_{period}_{today}.pkl"
            
            if os.path.exists(api_cache_file):
                with open(api_cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.debug(f"API 캐시 데이터 사용: {api_cache_file}")
                return cached_data
            
            return None
            
        except Exception as e:
            logger.error(f"API 캐시 조회 오류: {e}", exc_info=True)
            return None
    
    def _cache_api_data(self, code: str, period: str, dataframe: pd.DataFrame):
        """
        API 데이터를 캐시에 저장 (장 마감 후에만)
        
        Args:
            code: 종목 코드
            period: 조회 기간
            dataframe: 저장할 데이터
        """
        try:
            # 장중에는 API 데이터 캐싱하지 않음 (실시간 데이터 우선)
            if self.is_market_open():
                logger.debug(f"장중이므로 API 데이터 캐싱 생략: {code}")
                return
            
            today = datetime.now().strftime('%Y-%m-%d')
            api_cache_file = f"{self.api_cache_dir}/{code}_{period}_{today}.pkl"
            
            with open(api_cache_file, 'wb') as f:
                pickle.dump(dataframe, f)
            
            logger.debug(f"API 데이터 캐시 저장: {api_cache_file}, {len(dataframe)}개 레코드")
            
            # 이전 날짜의 API 캐시 파일들 정리
            self._cleanup_old_api_cache(code, period)
            
        except Exception as e:
            logger.error(f"API 데이터 캐시 저장 오류: {e}", exc_info=True)
    
    def _cleanup_old_api_cache(self, code: str, period: str):
        """
        오래된 API 캐시 파일들 정리 (7일 이상 된 파일 삭제)
        
        Args:
            code: 종목 코드
            period: 조회 기간
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=7)
            pattern = f"{code}_{period}_"
            
            for file in os.listdir(self.api_cache_dir):
                if file.startswith(pattern) and file.endswith('.pkl'):
                    try:
                        # 파일명에서 날짜 추출: code_period_YYYY-MM-DD.pkl
                        date_str = file.replace(pattern, '').replace('.pkl', '')
                        file_date = datetime.strptime(date_str, '%Y-%m-%d')
                        
                        if file_date < cutoff_date:
                            old_file_path = os.path.join(self.api_cache_dir, file)
                            os.remove(old_file_path)
                            logger.debug(f"오래된 API 캐시 파일 삭제: {file}")
                    except ValueError:
                        continue
                        
        except Exception as e:
            logger.error(f"API 캐시 정리 오류: {e}", exc_info=True) 