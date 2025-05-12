"""
한국 주식 자동매매 - 설정 모듈
설정 파일 로드 및 상수 정의
"""

import yaml

# 설정 파일 로드
def load_config():
    """
    config.yaml 파일에서 설정 로드
    
    Returns:
        dict: 설정 정보
    """
    with open('config.yaml', encoding='UTF-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

# 전역 설정 로드
_cfg = load_config()

# 실전/모의 API 설정
USE_REALTIME_API = True  # 실전투자: True, 모의투자: False
is_prod_env = USE_REALTIME_API  # 실전 환경 여부 (websocket.py 호환용)

# API 관련 설정
APP_KEY_REAL = _cfg['APP_KEY_REAL']
APP_SECRET_REAL = _cfg['APP_SECRET_REAL']
URL_BASE_REAL = _cfg['URL_BASE_REAL']

APP_KEY_VTS = _cfg['APP_KEY_VTS']
APP_SECRET_VTS = _cfg['APP_SECRET_VTS']
URL_BASE_VTS = _cfg['URL_BASE_VTS']

# 현재 모드에 따른 API 키 설정
APP_KEY = APP_KEY_REAL if USE_REALTIME_API else APP_KEY_VTS
APP_SECRET = APP_SECRET_REAL if USE_REALTIME_API else APP_SECRET_VTS
URL_BASE = URL_BASE_REAL if USE_REALTIME_API else URL_BASE_VTS

# 계좌 관련 설정
CANO_REAL = _cfg['CANO_REAL']
CANO_VTS = _cfg['CANO_VTS']
ACNT_PRDT_CD = _cfg['ACNT_PRDT_CD']

# 현재 모드에 따른 계좌 설정
CANO = CANO_REAL if USE_REALTIME_API else CANO_VTS

# 알림 관련 설정
DISCORD_WEBHOOK_URL = _cfg['DISCORD_WEBHOOK_URL']

# 웹소켓 URL 설정
WS_URL_REAL = "wss://ops.koreainvestment.com:21000/tryitout/H0STCNT0"
WS_URL_VTS = "wss://ops.koreainvestment.com:31000/tryitout/H0STCNT0"
WS_URL = WS_URL_REAL if USE_REALTIME_API else WS_URL_VTS

# 거래 관련 설정
TRADE_CONFIG = {
    "target_buy_count": 4,         # 매수할 종목 수
    "buy_percentage": 0.25,        # 종목당 매수 금액 비율
    "macd_short": 5,               # MACD 단기 기간
    "macd_long": 60,               # MACD 장기 기간
    "moving_avg_period": 20,       # 이동평균선 기간
    
    # 위험 관리 관련 설정
    "daily_loss_limit": 500000,    # 일일 최대 손실 제한 (원) - 절대값
    "daily_loss_limit_pct": 5.0,   # 일일 최대 손실 제한 (총 자산의 %)
    "daily_profit_limit_pct": 5.0, # 일일 최대 수익 제한 (총 자산의 %)
    "position_loss_limit": 50000,  # 단일 포지션 최대 손실 제한 (원)
    "max_position_size": 1000,     # 최대 포지션 크기 (주)
    "trailing_stop_pct": 3.0,      # 트레일링 스탑 비율 (%)
    # "max_day_trade_count": 10,   # 일일 최대 거래 횟수 - 제거됨
    "exposure_limit_pct": 70       # 최대 노출도 비율 (%)
}

# 관심 종목 필터링 기준
STOCK_FILTER = {
    "trade_volume_threshold": 1_000_000,          # 100만 주 이상
    "price_threshold": 3_000,                     # 현재가 3,000원 이상
    "market_cap_threshold": 400_000_000_000,      # 4,000억 원 이상
    "monthly_volatility_threshold": 10,           # 10개월 등락률 평균 ±10% 이내
    "trade_volume_increase_ratio": 4.0,           # 거래량 400% 증가 조건
    "close_price_increase_ratio": 0.05,           # 종가 5% 이상 상승 조건
    "score_threshold": 2                          # 최소 점수 기준
} 