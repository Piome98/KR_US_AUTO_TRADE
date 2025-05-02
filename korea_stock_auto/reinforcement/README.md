# 한국 주식 자동매매 - 강화학습 모듈

강화학습을 활용한 한국 주식 자동매매 기능을 제공하는 모듈입니다.

## 디렉토리 구조

```
korea_stock_auto/reinforcement/
├── data/                 # 데이터 처리 관련 모듈
│   ├── data_fetcher.py   # 데이터 수집 기능
│   ├── data_processor.py # 데이터 전처리 기능
│   └── __init__.py
│
├── models/               # 강화학습 모델 정의
│   ├── rl_model.py       # 모델 클래스 정의
│   └── __init__.py
│
├── training/             # 모델 학습 관련 모듈
│   ├── trainer.py        # 학습 파이프라인 구현
│   └── __init__.py
│
├── utils/                # 유틸리티 함수
│   ├── model_utils.py    # 모델 관련 유틸리티
│   └── __init__.py
│
├── main.py               # 메인 진입점
└── __init__.py
```

## 주요 기능

1. **데이터 처리**
   - `data_fetcher.py`: 주식 데이터 수집 (데이터베이스, Yahoo Finance 등)
   - `data_processor.py`: 기술적 지표 추가, 정규화, 시퀀스 생성

2. **강화학습 모델**
   - `rl_model.py`: 트레이딩 환경(TradingEnvironment), 모델 인터페이스(RLModel), 앙상블(ModelEnsemble) 정의
   - Stable Baselines3 기반의 다양한 알고리즘 지원 (PPO, A2C, DQN)

3. **학습 및 평가**
   - `trainer.py`: 모델 학습, 평가, 저장 파이프라인
   - 성능 시각화, 백테스트, 앙상블 생성 기능

4. **유틸리티**
   - `model_utils.py`: 모델 관리, 비교, 시각화 유틸리티

## 사용 방법

### 모델 학습

```bash
python -m korea_stock_auto.reinforcement.main --mode train --code 005930 --model-type ppo --timesteps 100000
```

### 모델 테스트

```bash
python -m korea_stock_auto.reinforcement.main --mode test --code 005930 --model-id ppo_20230501_120000
```

### 백테스트

```bash
python -m korea_stock_auto.reinforcement.main --mode backtest --code 005930 --model-id ppo_20230501_120000 --start-date 2023-01-01 --end-date 2023-12-31
```

### 앙상블 생성

```bash
python -m korea_stock_auto.reinforcement.main --mode ensemble --top-n 3
```

### 모델 비교

```bash
python -m korea_stock_auto.reinforcement.main --mode compare --code 005930 --models ppo_20230501_120000 a2c_20230502_120000
```

### 모델 목록 조회

```bash
python -m korea_stock_auto.reinforcement.main --mode list
```

## 매개변수 설명

- `--mode`: 실행 모드 (train, test, ensemble, backtest, compare, list)
- `--code`: 종목 코드
- `--start-date`: 시작 날짜 (YYYY-MM-DD)
- `--end-date`: 종료 날짜 (YYYY-MM-DD)
- `--model-type`: 강화학습 모델 유형 (ppo, a2c, dqn)
- `--timesteps`: 학습 스텝 수
- `--output-dir`: 모델 저장 디렉토리
- `--model-id`: 테스트할 모델 ID
- `--ensemble`: 앙상블 모델 사용 여부 (플래그)
- `--top-n`: 앙상블에 포함할 상위 모델 개수
- `--models`: 비교할 모델 ID 목록 