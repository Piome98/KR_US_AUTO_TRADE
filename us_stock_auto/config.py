"""
미국 주식 자동매매 - 설정 모듈
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

# API 관련 설정
APP_KEY = _cfg['APP_KEY']
APP_SECRET = _cfg['APP_SECRET']
URL_BASE = _cfg['URL_BASE']

# 계좌 관련 설정
CANO = _cfg['CANO']
ACNT_PRDT_CD = _cfg['ACNT_PRDT_CD']

# 알림 관련 설정
DISCORD_WEBHOOK_URL = _cfg['DISCORD_WEBHOOK_URL']

# 거래소 코드 매핑
EXCHANGE_CODE_MAP = {
    # API용 시장 코드
    "NASDAQ": "NASD",  
    "NYSE": "NYSE",
    "AMEX": "AMEX",
    # 시세 조회용 시장 코드
    "NASDAQ_PRICE": "NAS",
    "NYSE_PRICE": "NYS",
    "AMEX_PRICE": "AMS"
}

# 매매 설정
TRADE_CONFIG = {
    "target_buy_count": 3,    # 매수할 종목 수
    "buy_percent": 0.33       # 종목당 매수 금액 비율
}

# 타임존 설정
TIMEZONE = 'America/New_York'

# 시간 설정
MARKET_TIME = {
    "open": (9, 30),          # 장 시작 시간 (9:30)
    "buy_start": (9, 35),     # 매수 시작 시간 (9:35)
    "sell_start": (15, 45),   # 매도 시작 시간 (15:45)
    "exit": (15, 50)          # 프로그램 종료 시간 (15:50)
} 