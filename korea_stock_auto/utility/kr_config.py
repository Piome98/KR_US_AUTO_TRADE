# 설정 관리 파일 -> API KEY 및 API SECRET 등 개인 정보 관리
# korea_stock_auto.utility.real_utils.kr_config


import yaml

# YAML 파일 로드
with open('config.yaml', encoding='utf-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)

# API 설정(실전)
APP_KEY_REAL = _cfg["APP_KEY_REAL"]
APP_SECRET_REAL = _cfg["APP_SECRET_REAL"]
URL_BASE_REAL = _cfg["URL_BASE_REAL"]

# API 설정(모의)
APP_KEY_VTS = _cfg["APP_KEY_VTS"]
APP_SECRET_VTS = _cfg["APP_SECRET_VTS"]
URL_BASE_VTS = _cfg["URL_BASE_VTS"]


# 계좌 설정
CANO_REAL = _cfg["CANO_REAL"]
CANO_VTS = _cfg["CANO_VTS"]
ACNT_PRDT_CD = _cfg["ACNT_PRDT_CD"]

# 디스코드 웹훅
DISCORD_WEBHOOK_URL = _cfg["DISCORD_WEBHOOK_URL"]

# 웹소켓 실전/모의 분리
USE_REALTIME_API = True #모의투자 기본값, 실전투자는 True로 변경

APP_KEY = APP_KEY_REAL if USE_REALTIME_API else APP_KEY_VTS
APP_SECRET = APP_SECRET_REAL if USE_REALTIME_API else APP_SECRET_VTS
URL_BASE = URL_BASE_REAL if USE_REALTIME_API else URL_BASE_VTS