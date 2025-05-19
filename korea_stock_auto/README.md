# 한국 주식 자동매매 시스템 (Korea Stock Auto Trading System)

## 개요

한국투자증권 API를 활용하여 한국 주식 시장에서 자동으로 매매를 수행하는 시스템입니다. 모듈화된 구조로 설계되어 유지보수 및 확장이 용이합니다.

## 최근 변경사항

### ETF 필터링 및 호가 조회 기능 개선 (2025-05-19)

ETF 필터링 및 호가 조회 관련 기능을 개선했습니다:

- **ETF 종목 필터링 기능**:
  - `is_etf_stock()` 함수 추가하여 ETF 종목 판별 기능 제공
  - ETF 종목을 관심종목에서 제외하는 옵션 추가 (`EXCLUDE_ETF` 설정)
  - 시장 구분 코드 'J'(주식)와 'E'(ETF)를 적절히 구분하여 API 호출

- **호가 조회 기능 수정**:
  - 예상 체결가 호가 조회 관련 오류를 해결하기 위해 호가 조회 기능 제거
  - 현재가 조회 결과만으로 매매에 필요한 정보를 제공하도록 기능 개선
  - `get_real_time_price_by_api()` 함수가 현재가 정보만 사용하도록 수정

- **개선 효과**:
  - API 호출 성공률 향상
  - ETF와 개별 주식 종목 간 구분 명확화
  - 호가 정보 조회 실패로 인한 매매 오류 방지

### 강화학습 모듈 모듈화 작업 (2025-05-13)

강화학습 관련 모듈들을 더 작은 단위로 모듈화하여 코드 구조를 개선했습니다:

- **rl_models 디렉토리 모듈화**:
  - 기존 `rl_model.py` 파일을 여러 모듈로 분리
  - `environment.py`: TradingEnvironment 클래스 정의
  - `model.py`: RLModel 클래스 정의
  - `ensemble.py`: ModelEnsemble 클래스 정의
  - `__init__.py`: 통합 인터페이스 제공

- **training 디렉토리 모듈화**:
  - 기존 `trainer.py` 파일을 여러 모듈로 분리
  - `base_trainer.py`: 기본 트레이너 클래스와 공통 기능
  - `callbacks.py`: 콜백 관련 코드 (SaveOnBestTrainingRewardCallback, EarlyStoppingCallback)
  - `evaluation.py`: 모델 평가 관련 기능 (ModelEvaluator)
  - `pipeline.py`: 전체 학습 파이프라인 기능 (TrainingPipeline)
  - `__init__.py`: 통합 인터페이스와 간소화된 train_model 함수 제공

- **rl_data와 data/database 모듈 통합**:
  - 중복 기능 제거 및 통합
  - `technical_indicators.py`와 `rl_data_manager.py` 파일을 리팩토링하여 `data/database` 모듈 활용

- **개선 효과**:
  - 코드 가독성 및 유지보수성 향상
  - 기능별 모듈 분리로 각 컴포넌트의 책임 명확화
  - 확장성 및 테스트 용이성 향상
  - 코드 재사용 및 중복 제거

### 웹소켓 모듈 모듈화 작업 (2025-05-12)

웹소켓 관련 코드를 모듈화하여 코드 구조를 개선했습니다:

- **웹소켓 모듈 분리**:
  - 기존 `websocket.py` 파일을 여러 모듈로 분리하여 단일 책임 원칙 준수
  - `connection_manager.py`: 웹소켓 연결 관리 기능
  - `subscription_manager.py`: 종목 구독/해지 관리 기능
  - `data_processor.py`: 실시간 시세 데이터 처리 기능
  - `stock_websocket.py`: 모든 컴포넌트를 통합하는 메인 클래스

- **순환 참조 문제 해결**:
  - `TYPE_CHECKING` 방식을 사용하여 모든 모듈에서 순환 참조 문제 해결
  - 타입 힌트를 위한 임포트는 런타임에 실행되지 않도록 처리

- **개선 효과**:
  - 코드 가독성 및 유지보수성 향상
  - 기능별 모듈 분리로 테스트 용이성 증가
  - 각 컴포넌트의 책임 명확화
  - 확장성 향상

### API 클라이언트 모듈 추가 확장 (2025-05-16)

API 클라이언트 모듈에 새로운 기능을 추가했습니다:

- **새로운 API 모듈 추가**:
  - `market/balance_sheet.py`: 기업 재무상태표 조회 기능
  - `market/basic_info.py`: 기업 기본 정보 조회 기능
  - `market/financial_ratios.py`: 기업 재무비율 조회 기능

- **개선 효과**:
  - 기업 재무 데이터 활용 가능
  - 펀더멘털 분석을 위한 데이터 접근성 향상
  - 강화학습 모델에 재무 데이터 통합 가능

### 강화학습 데이터 처리 코드 리팩토링 (2025-05-15)

강화학습 데이터 처리 모듈을 리팩토링하여 코드 구조를 개선했습니다:

- **데이터 처리 모듈 분리**:
  - 기존 `data_processor.py` (647줄)를 여러 모듈로 분리하여 단일 책임 원칙 준수
  - `data_preprocessor.py`: 기본 데이터 로딩 및 정제 기능
  - `technical_indicators.py`: 기술적 지표 계산 기능
  - `data_normalizer.py`: 데이터 정규화 기능
  - `sequence_generator.py`: 시계열 시퀀스 생성 기능
  - `market_data_integrator.py`: 시장 데이터 통합 기능
  - `rl_data_manager.py`: 전체 데이터 파이프라인 관리 기능

- **수정 효과**:
  - 코드 가독성 및 유지보수성 향상
  - 개별 데이터 처리 기능에 대한 단위 테스트 용이
  - 로직 분리를 통한 코드 재사용성 강화
  - 새로운 데이터 처리 기능 추가가 용이해짐

### API 디렉토리 중복 기능 개선 (2025-05-12)

API 디렉토리의 중복 기능을 확인하고 개선했습니다:

- **중복 함수 통합**:
  - `fetch_stock_current_price()`와 `get_real_time_price_by_api()` 함수의 중복 기능 통합
  - `get_real_time_price_by_api()` 함수를 직접 API를 호출하도록 수정
  - `fetch_stock_current_price()` 함수는 내부적으로 `get_real_time_price_by_api()`를 호출하도록 변경

- **개선 효과**:
  - 코드 중복 제거
  - API 호출 일관성 향상
  - 기능 확장 시 한 곳에서만 수정 필요
  - 유지보수성 향상

### 코드 오류 수정 및 종속성 추가 (2025-05-12)

강화학습 모듈의 오류를 수정하고 종속성을 추가했습니다:

- **누락된 임포트 추가**:
  - `rl_models/rl_model.py`에 `pandas` 임포트 추가
  - `training/trainer.py`에 `pickle` 임포트 추가
  - `training/trainer.py`에 `select_features`와 `get_state_dim` 함수 임포트 추가

- **오류 수정 효과**:
  - 강화학습 모델의 학습 및 예측 기능이 정상적으로 작동
  - 특성 선택 모듈과의 연동 강화
  - 모델 저장 및 로드 기능 정상화

### API 데이터 활용 및 캐싱 메커니즘 강화 (2025-05-11)

API를 통한 데이터 수집 및 캐싱 메커니즘을 강화했습니다:

- **API 호출 기능 강화**:
  - `get_volume_increasing_stocks()`: 거래량 급증 종목 조회 API 함수 추가
  - `get_real_time_price_by_api()`: 현재가 호가 통합 조회 함수 추가
  - API 데이터를 강화학습에 활용할 수 있도록 전처리 기능 추가

- **데이터 캐싱 메커니즘 구현**:
  - API 호출 결과를 로컬 JSON 파일로 저장
  - 시간 기반 유효성 검사 적용 (현재가: 1시간, 거래량 순위: 24시간)
  - API 장애 시 캐시 데이터 활용으로 안정성 향상

- **강화학습 모듈 개선**:
  - API 데이터를 강화학습에 활용할 수 있도록 특성화
  - `api_enhanced` 특성 세트 추가 (기본 특성 + API 데이터 특성)
  - 시장 압력, 매수/매도 비율 등 API 데이터 기반 특성 활용

### API 클라이언트 모듈 리팩토링 (2025-05-10)

API 클라이언트(`api_client.py`) 파일을 기능별로 세분화하여 모듈화했습니다:

- **모듈화된 구조**:
  - `base/client.py`: 기본 API 클라이언트 클래스 정의
  - `account/balance.py`: 계좌 잔고 관련 기능
  - `account/deposit.py`: 계좌 예수금 관련 기능
  - `market/price.py`: 시장 가격 조회 기능
  - `market/chart.py`: 차트 데이터 조회 기능
  - `market/stock_info.py`: 주식 기본 정보 조회 기능
  - `order/stock.py`: 주식 주문 관련 기능
  - `order/status.py`: 주문 상태 조회 기능
  - `sector/index.py`: 업종 지수 관련 기능
  - `sector/info.py`: 업종 정보 관련 기능

- **주요 개선점**:
  - 각 기능이 Mixin 클래스로 구현되어 `KoreaInvestmentApiClient` 클래스가 모든 기능을 상속
  - 기존 2500라인에 달하던 파일을 기능별로 분리하여 가독성 및 유지보수성 향상
  - 기능 확장 시 기존 코드 수정 없이 새 Mixin 클래스 추가 가능
  - 각 모듈별 단위 테스트가 용이하도록 구조 개선

### 업종시세 관련 기능 추가 (주석 처리) (2025-05-07)

국내 업종시세 관련 기능을 주석 처리된 상태로 추가했습니다. 필요할 때 주석을 해제하여 사용할 수 있습니다:

- 업종 조회 기능:
  - `get_sector_current_index()`: 국내업종 현재지수 조회 (코스피, 코스닥 및 세부 업종 지수)
  - `get_sector_daily_index()`: 국내업종 일자별지수 조회
  - `get_sector_minute_chart()`: 업종 분봉 데이터 조회
  - `get_sector_period_index()`: 업종 기간별 시세 조회 (일/주/월/년)
  - `get_sector_category_stocks()`: 국내업종 구분별 전체시세 조회 (특정 업종에 속한 종목들의 시세)

- 각 업종별 상세 데이터 제공:
  - 업종 현재지수, 등락률, 거래량, 시가총액 등 기본 정보
  - 상승/하락/보합 종목 수 및 비율
  - 52주 신고가/저가 정보
  - 특정 업종 내 개별 종목의 시세 데이터

### 주식 시세 관련 기능 강화 (2025-05-06)

주식 시세 조회 관련 기능을 대폭 확장했습니다:

- 시세 조회 기능 강화:
  - `get_stock_asking_price()`: 주식 현재가 호가 정보 조회 기능 추가
  - `get_stock_conclusion()`: 주식 현재가 체결 정보 조회 기능 추가
  - `get_stock_time_conclusion()`: 주식 당일 시간대별 체결 내역 조회 기능 추가
  - `get_stock_minute_data()`: 주식 당일 분봉 데이터 조회 기능 추가
  - `get_investor_trends()`: 주식 현재가 투자자별 매매현황 조회 기능 추가

- 기존 시세 조회 기능 개선:
  - `fetch_stock_current_price()`: 현재가 조회 결과에 52주 신고가/저가, PER, EPS, PBR 등 상세 정보 추가
  - 시세 데이터를 보다 사용하기 쉬운 형태로 가공하여 반환하도록 개선
  - 모든 시세 함수에 모의투자 지원 여부 명시

### API 모듈 주식 주문 및 계좌 관리 기능 강화 (2025-05-05)

주식 주문 및 계좌 관리와 관련된 API 기능을 대폭 개선했습니다:

- 계좌 관련 기능 강화:
  - `get_account_info()`: 계좌 기본 정보 조회 추가
  - `fetch_buyable_amount()`: 매수 가능 금액 조회 기능 추가
  - `fetch_sellable_quantity()`: 매도 가능 수량 조회 기능 추가
  - `get_buyable_cash()`: 주문 가능 현금 조회 기능 추가
  - `get_profit_loss()`: 기간별 손익 조회 기능 추가 (실전투자만)
  - `get_realized_profit_loss()`: 실현손익 조회 기능 추가 (실전투자만)

- 주문 관련 기능 강화:
  - `modify_order()`: 주문 정정 기능 추가
  - `get_pending_orders()`: 미체결 주문 조회 기능 추가 (실전투자만)
  - `get_executed_orders()`: 체결 내역 조회 기능 추가
  - `cancel_all_orders()`: 모든 미체결 주문 일괄 취소 기능 추가 (실전투자만)

- 에러 처리 및 안정성 개선:
  - API 호출 속도 제한(Rate Limiting) 준수
  - API 응답 오류 처리 강화
  - 토큰 만료 시 자동 갱신 기능 개선
  - 모의투자/실전투자 환경 구분 및 지원 여부 명시

### 디렉토리 구조 리팩토링 (2025-05-02)

기능별로 파일들을 분류하여 더 체계적인 디렉토리 구조로 변경했습니다:

- 기존 평면적 구조에서 모듈화된 디렉토리 구조로 개선
- 유사한 기능을 하는 파일들을 같은 디렉토리에 배치
- 중복 이름 충돌 방지를 위해 reinforcement 디렉토리 내 하위 디렉토리 이름 변경:
  - models → rl_models
  - data → rl_data
  - utils → rl_utils

## 디렉토리 구조

### 1. 메인 디렉토리

- **korea_stock_auto/**: 프로젝트 루트 디렉토리
  - **api/**: API 통신 관련 모듈
  - **data/**: 데이터베이스 및 데이터 관리 모듈
  - **models/**: 모델 관리 모듈
  - **trading/**: 트레이딩 관련 모듈
  - **utils/**: 유틸리티 함수
  - **reinforcement/**: 강화학습 관련 모듈
    - **rl_data/**: 강화학습용 데이터 전처리 모듈
    - **rl_models/**: 강화학습 모델 모듈
    - **rl_utils/**: 강화학습 유틸리티 모듈
    - **training/**: 모델 학습 모듈
  - **main.py**: 메인 실행 파일
  - **config.py**: 설정 관리 및 상수 정의

## 모듈 역할

### 1. API 모듈 (api/)

- **api_client/**: 한국투자증권 API 클라이언트 (모듈화됨)
  - **base/**: 기본 클래스 및 유틸리티
    - **client.py**: 기본 API 클라이언트 클래스 정의
  - **account/**: 계좌 관련 기능
    - **balance.py**: 계좌 잔고 관련 기능
    - **deposit.py**: 계좌 예수금 관련 기능
  - **market/**: 시장 데이터 관련 기능
    - **price.py**: 시장 가격 조회 기능
    - **chart.py**: 차트 데이터 조회 기능
    - **stock_info.py**: 주식 기본 정보 조회 기능
    - **balance_sheet.py**: 기업 재무상태표 조회 기능
    - **basic_info.py**: 기업 기본 정보 조회 기능
    - **financial_ratios.py**: 기업 재무비율 조회 기능
  - **order/**: 주문 관련 기능
    - **stock.py**: 주식 주문 관련 기능
    - **status.py**: 주문 상태 조회 기능
  - **sector/**: 업종 관련 기능
    - **index.py**: 업종 지수 관련 기능
    - **info.py**: 업종 정보 관련 기능
- **auth.py**: API 인증 및 토큰 관리
- **websocket/**: 실시간 시세 수신을 위한 웹소켓 클라이언트 (모듈화됨)
  - **connection_manager.py**: 웹소켓 연결 관리 기능
  - **subscription_manager.py**: 종목 구독/해지 관리 기능
  - **data_processor.py**: 실시간 시세 데이터 처리 기능
  - **stock_websocket.py**: 모든 컴포넌트를 통합하는 메인 클래스

### 2. 데이터 모듈 (data/)

- **database.py**: 데이터베이스 연동 및 쿼리 실행

### 3. 모델 모듈 (models/)

- **model_manager.py**: 모델 관리 인터페이스
- **models.py**: 기본 모델 클래스 정의

### 4. 트레이딩 모듈 (trading/)

- **trader.py**: 매매 로직 및 전략 구현

### 5. 유틸리티 모듈 (utils/)

- **utils.py**: 공통 유틸리티 함수 모음
- **db_helper.py**: 데이터베이스 연결 헬퍼 함수

### 6. 강화학습 모듈 (reinforcement/)

- **rl_data/**: 강화학습 데이터 처리 모듈 (리팩토링 완료)
  - **data_fetcher.py**: 데이터 수집 및 가져오기
  - **data_preprocessor.py**: 기본 데이터 로딩 및 정제
  - **technical_indicators.py**: 기술적 지표 계산
  - **data_normalizer.py**: 데이터 정규화
  - **sequence_generator.py**: 시계열 시퀀스 생성
  - **market_data_integrator.py**: 시장 데이터 통합
  - **rl_data_manager.py**: 전체 데이터 파이프라인 관리
- **rl_models/**: 강화학습 모델 모듈 (모듈화 완료)
  - **environment.py**: OpenAI Gym 기반 트레이딩 환경
  - **model.py**: 강화학습 모델 래퍼 클래스
  - **ensemble.py**: 모델 앙상블 클래스
  - **features.py**: 특성 선택 및 처리 기능
  - **__init__.py**: 통합 인터페이스
- **rl_utils/model_utils.py**: 모델 관리 유틸리티
- **training/**: 모델 학습 및 평가 모듈 (모듈화 완료)
  - **base_trainer.py**: 기본 트레이닝 기능 제공
  - **callbacks.py**: 콜백 관련 코드
  - **evaluation.py**: 모델 평가 관련 기능
  - **pipeline.py**: 전체 학습 과정 관리
  - **__init__.py**: 통합 인터페이스와 간소화된 함수
- **main.py**: 강화학습 메인 실행 파일

## 강화학습 모듈 상세 설명

### 1. 데이터 전처리 (rl_data/ 모듈)

강화학습 데이터 전처리 모듈은 다음과 같이 분리되었습니다:

- **data_preprocessor.py**: 기본 데이터 로딩, 정제 및 전처리
- **technical_indicators.py**: 기술적 지표 계산 및 추가
- **data_normalizer.py**: 데이터 정규화
- **sequence_generator.py**: 시계열 시퀀스 생성
- **market_data_integrator.py**: 시장 데이터 통합
- **rl_data_manager.py**: 전체 데이터 파이프라인 관리

**주요 기능:**

- **기술적 지표 추가**: 주가 데이터에 기술적 지표(이동평균선, RSI, MACD, 볼린저 밴드 등)를 추가합니다.
- **데이터 정규화**: 서로 다른 스케일을 가진 데이터를 0~1 범위로 정규화하여 학습 효율을 높입니다.
- **시계열 시퀀스 생성**: 과거 n일치 데이터를 묶어 시계열 시퀀스를 생성합니다.
- **상태 벡터 준비**: 현재 시장 상태를 모델이 이해할 수 있는 벡터 형태로 변환합니다.

**데이터 가공 과정:**

1. 원시 OHLCV(시가, 고가, 저가, 종가, 거래량) 데이터 로드
2. 기술적 지표 계산 및 추가
3. 이상치와 결측값 처리
4. MinMaxScaler를 사용하여 데이터 정규화
5. 학습에 사용할 시퀀스 데이터 생성
6. 예측에 사용할 현재 상태 벡터 생성

### 2. 강화학습 모델 (rl_models/)

강화학습 모델 모듈은 트레이딩 환경 정의와 모델 구현을 담당합니다.

**주요 클래스:**

- **TradingEnvironment** (`environment.py`): OpenAI Gym 기반의 트레이딩 환경
- **RLModel** (`model.py`): 강화학습 모델 인터페이스
- **ModelEnsemble** (`ensemble.py`): 여러 모델을 앙상블하는 클래스

**TradingEnvironment 작동 원리:**

1. 환경은 주가 데이터와 에이전트의 포트폴리오 상태를 관리합니다.
2. 액션 공간: 0(홀드), 1(매수), 2(매도)
3. 관찰 공간: 주가 데이터 + 기술적 지표 + 포트폴리오 상태
4. 보상 함수: 거래 완료 시 수익률 기반 보상 + 홀딩 패널티

**지원 모델 유형:**

- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- DQN (Deep Q-Network)

**앙상블 메커니즘:**

- 여러 모델의 예측을 가중치에 따라 결합하여 최종 결정 도출
- 성과가 좋은 모델에 더 높은 가중치 부여
- 다양한 모델 조합으로 예측 안정성 향상

### 3. 모델 학습 및 평가 (training/)

모델 학습 및 평가 모듈은 강화학습 모델의 학습과 성능 평가를 담당합니다.

**주요 컴포넌트:**

- **BaseTrainer** (`base_trainer.py`): 기본 트레이닝 기능 제공
- **ModelEvaluator** (`evaluation.py`): 학습된 모델의 성능 평가
- **TrainingPipeline** (`pipeline.py`): 전체 학습 과정 관리
- **각종 콜백** (`callbacks.py`): 학습 중 특정 상황에서 동작하는 콜백 함수들

**주요 기능:**

- **데이터 로드**: 데이터베이스에서 학습 데이터 로드
- **데이터셋 준비**: 학습/테스트 데이터 분할 및 전처리
- **모델 학습**: 다양한 파라미터로 모델 학습
- **모델 평가**: 백테스트를 통한 모델 성능 평가
- **결과 시각화**: 성능 지표 및 트레이딩 결과 시각화

**TrainingPipeline 작동 원리:**

1. 데이터 로드 및 전처리
2. 학습/테스트 데이터 분할
3. 트레이딩 환경 생성
4. 모델 학습 수행
5. 테스트 데이터로 모델 평가
6. 성능 지표 계산(수익률, 샤프 비율, 최대 낙폭, 승률)
7. 결과 저장 및 시각화

**앙상블 모델 생성 기능:**

- 과거 학습된 모델 중 성능 상위 n개 선택
- 각 모델의 가중치 설정 (성능에 비례)
- 앙상블 설정 저장 및 재사용 가능

## 데이터 가공 방법

강화학습 모델에 사용할 데이터 가공은 다음 단계로 이루어집니다:

1. **원시 데이터 수집**:

   - 일별/분별 OHLCV 데이터 수집
   - 필요시 기업 재무정보, 시장 지표 등 추가 데이터 수집

2. **특성 엔지니어링**:

   - 기술적 지표 계산 추가 (이동평균선, RSI, MACD, 볼린저 밴드 등)
   - 가격 모멘텀, 변동성 지표 계산
   - 날짜 기반 특성 추가 (요일, 월말효과 등)

3. **데이터 정규화**:

   - MinMaxScaler를 사용하여 특성을 0~1 범위로 정규화
   - 학습/테스트 데이터 누수 방지를 위한 스케일러 관리

4. **상태 표현 구성**:

   - 시계열 데이터 윈도우(lookback) 설정
   - 주가 패턴, 기술적 지표, 포지션 정보를 포함한 상태 벡터 구성
   - 강화학습 에이전트가 인식할 수 있는 고정 크기 관찰 공간 정의

5. **보상 함수 설계**:
   - 거래 완료 시 실현 수익률 기반 보상
   - 홀딩 시 미실현 손익 기반 보상
   - 장기 홀딩 방지를 위한 패널티 적용
   - 리스크 관리 요소 통합 (최대 낙폭 패널티 등)

## 사용 가능한 강화학습 모델

현재 지원되는 강화학습 모델과 특징:

1. **PPO (Proximal Policy Optimization)**:

   - 안정적인 학습 성능
   - 샘플 효율성이 높음
   - 하이퍼파라미터 조정이 상대적으로 쉬움
   - `learning_rate`, `n_steps`, `gamma` 조정 가능

2. **A2C (Advantage Actor-Critic)**:

   - 병렬 환경에서 효율적
   - 빠른 학습 속도
   - 액터-크리틱 구조로 안정성 향상
   - `learning_rate`, `n_steps`, `ent_coef` 조정 가능

3. **DQN (Deep Q-Network)**:
   - 이산 액션 공간에 적합
   - 경험 리플레이를 통한 샘플 효율성
   - 타겟 네트워크로 안정성 향상
   - `learning_rate`, `buffer_size`, `exploration_fraction` 조정 가능

사용자는 필요에 따라 모델 유형을 선택하고, 하이퍼파라미터를 조정하여 학습 성능을 최적화할 수 있습니다.

## 향후 개발 계획

1. **API 모듈 개선**:
   - REST API와 웹소켓 통신 견고성 강화
   - 실시간 데이터 처리 기능 개선
   - 다양한 증권사 API 지원 확장

2. **데이터 모듈 확장**:
   - 데이터 저장 및 캐싱 전략 개선
   - 기술적 지표 라이브러리 확장
   - 실시간 데이터 통합 관리

3. **모델 모듈 고도화**:
   - 다양한 모델 추가 (시계열, 회귀, 분류 등)
   - 모델 검증 및 평가 프레임워크 강화
   - 하이퍼파라미터 최적화 자동화

4. **트레이딩 모듈 개선**:
   - 다양한 트레이딩 전략 구현
   - 위험 관리 및 포트폴리오 최적화
   - 주문 실행 알고리즘 개선

5. **강화학습 모듈 확장**:
   - 최신 강화학습 알고리즘 추가 (SAC, TD3, DDPG 등)
   - 멀티에이전트 학습 도입
   - 설명 가능한 AI 기법 통합


2025-05-15:
# __pycache__ 디렉토리 삭제
Get-ChildItem -Path . -Include __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force

# 프로그램 실행
python -m korea_stock_auto.main --cycles 1

## API 버그 수정 내역

### 거래량 상위 종목 조회 API 문제 해결 (2025-05-17)

한국투자증권 API를 통한 거래량 상위 종목 조회 기능의 문제를 수정했습니다:

- **문제 원인**:
  - `FID_COND_SCR_DIV_CODE` 파라미터 값이 올바르지 않았음
  - 거래량 순위 조회 API 요청 시 필요한 파라미터 값 부족

- **수정 내용**:
  - 거래량 순위 조회 API 파라미터를 국내주식-047 문서 기준으로 수정:
    - `FID_COND_SCR_DIV_CODE` 값을 `20171`로 설정
    - `FID_TRGT_EXLS_CLS_CODE` 값을 `0000000000`으로 설정
    - 기타 필수 파라미터 추가 및 수정

- **개선 효과**:
  - 거래량 상위 종목 API 정상 작동
  - 거래량 급증 종목 API 동일하게 적용하여 정상화
  - 관심종목 선정 프로세스 정상 진행
  - 거래량 상위 종목에 기반한 트레이딩 전략 정상 작동

- **기술적 상세**:
  - API 문서 `거래량순위[v1_국내주식-047].xlsx` 기준으로 파라미터 검토
  - 파라미터 값 수정 및 누락된 파라미터 추가
  - 헤더 부분의 트레이딩 ID 설정 확인
  - 문제 해결을 위해 __pycache__ 초기화 적용