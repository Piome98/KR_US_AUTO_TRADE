"""
강화학습 자동매매 메인 모듈
"""

import os
import argparse
import datetime
import pandas as pd
import json
from korea_stock_auto.utils import send_message
from korea_stock_auto.data.database import DatabaseManager
from korea_stock_auto.reinforcement.rl_data.data_fetcher import DataFetcher
from korea_stock_auto.reinforcement.rl_data.rl_data_manager import RLDataManager
from korea_stock_auto.reinforcement.training.trainer import ModelTrainer, create_ensemble_from_best_models
from korea_stock_auto.reinforcement.rl_models.rl_model import RLModel, ModelEnsemble
from korea_stock_auto.reinforcement.rl_utils.model_utils import (
    list_available_models, 
    load_model_by_id, 
    compare_models, 
    visualize_model_performance
)


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='강화학습 자동매매 시스템')
    
    # 기본 설정
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'ensemble', 'backtest', 'compare', 'list'],
                        help='실행 모드 (train, test, ensemble, backtest, compare, list)')
    parser.add_argument('--code', type=str, default='005930',
                        help='종목 코드')
    parser.add_argument('--start-date', type=str, default=None,
                        help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='종료 날짜 (YYYY-MM-DD)')
    
    # 학습 설정
    parser.add_argument('--model-type', type=str, default='ppo',
                        choices=['ppo', 'a2c', 'dqn'],
                        help='강화학습 모델 유형')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='학습 스텝 수')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='모델 저장 디렉토리')
    
    # 테스트 설정
    parser.add_argument('--model-id', type=str, default=None,
                        help='테스트할 모델 ID')
    parser.add_argument('--ensemble', action='store_true',
                        help='앙상블 모델 사용 여부')
    
    # 앙상블 설정
    parser.add_argument('--top-n', type=int, default=3,
                        help='앙상블에 포함할 상위 모델 개수')
    
    # 비교 설정
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='비교할 모델 ID 목록')
    
    return parser.parse_args()


def train_model(args):
    """
    모델 학습
    
    Args:
        args: 명령행 인자
    """
    try:
        # 데이터베이스 연결
        db_manager = DatabaseManager()
        
        # 데이터 관리자 및 트레이너 생성
        data_manager = RLDataManager()
        trainer = ModelTrainer(
            db_manager=db_manager, 
            data_manager=data_manager,
            output_dir=args.output_dir
        )
        
        # 추가: 거래량 상위 종목 및 실시간 시세 데이터 수집 (모델 강화 목적)
        from korea_stock_auto.api.api_client.market.stock_info import StockInfoMixin
        from korea_stock_auto.api.api_client.market.price import MarketPriceMixin
        from korea_stock_auto.api.api_client.base.client import KoreaInvestmentApiClient
        from korea_stock_auto.reinforcement.rl_data.data_fetcher import DataFetcher
        
        # 거래량 상위 종목 정보 수집 (관련 종목 분석용)
        try:
            send_message(f"거래량 상위 종목 정보 수집 시작", config.notification.discord_webhook_url)
            api_client = KoreaInvestmentApiClient()
            
            # 일반 거래량 순위
            top_stocks = api_client.get_top_traded_stocks(market_type="0", top_n=10)
            
            # 거래량 급증 종목 정보 (신규)
            increasing_stocks = api_client.get_volume_increasing_stocks(market_type="0", top_n=10)
            
            # 주요 거래 종목 파악 및 관련 정보 출력
            if top_stocks:
                send_message(f"현재 거래량 상위 종목: {', '.join([s['name'] for s in top_stocks[:5]])}")
                
                # 거래량 상위 종목 관련 추가 데이터 수집 (대상 종목 포함 여부 확인)
                if args.code in [s['code'] for s in top_stocks]:
                    index = [s['code'] for s in top_stocks].index(args.code)
                    send_message(f"{args.code}는 현재 거래량 {index+1}위 종목입니다. 시장 관심도가 높습니다.", config.notification.discord_webhook_url)
            
            # 거래량 급증 종목 정보 출력
            if increasing_stocks:
                send_message(f"현재 거래량 급증 종목: {', '.join([s['name'] for s in increasing_stocks[:5]])}")
                
                # 거래량 급증 종목에 포함되어 있는지 확인
                if args.code in [s['code'] for s in increasing_stocks]:
                    index = [s['code'] for s in increasing_stocks].index(args.code)
                    ratio = increasing_stocks[index]['volume_ratio']
                    send_message(f"{args.code}는 현재 거래량 급증 {index+1}위 종목입니다. 증가율: {ratio}%", config.notification.discord_webhook_url)
                    
            # 데이터 캐싱 (시간 기록)
            cache_dir = os.path.join(os.path.dirname(__file__), '../../data/cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            # 학습 데이터로 활용하기 위한 저장
            with open(os.path.join(cache_dir, f'volume_rank_data.json'), 'w') as f:
                json.dump(top_stocks, f)
            
            with open(os.path.join(cache_dir, f'volume_increasing_data.json'), 'w') as f:
                json.dump(increasing_stocks, f)
                
        except Exception as e:
            send_message(f"거래량 상위 종목 정보 수집 실패: {e}", config.notification.discord_webhook_url)
        
        # 추가: 실시간 시세/호가 정보 수집 (학습 데이터 강화)
        try:
            # 실시간 통합 시세 정보 수집 (신규)
            real_time_info = api_client.get_real_time_price_by_api(code=args.code)
            
            if real_time_info:
                current_price = real_time_info.get("current_price", 0)
                change_rate = real_time_info.get("change_rate", 0)
                bid_ask_ratio = real_time_info.get("bid_ask_ratio", 0)
                
                send_message(f"{args.code} 현재가: {current_price}원 ({change_rate}%, config.notification.discord_webhook_url), 매수/매도 비율: {bid_ask_ratio:.2f}")
                
                # 시장 압력 지표 분석 (매수세/매도세 판단)
                if bid_ask_ratio > 1.2:  # 매수세가 강한 경우
                    send_message(f"매수세가 강한 상태입니다. 학습 모델에 이 정보가 반영됩니다.", config.notification.discord_webhook_url)
                elif bid_ask_ratio < 0.8:  # 매도세가 강한 경우
                    send_message(f"매도세가 강한 상태입니다. 학습 모델에 이 정보가 반영됩니다.", config.notification.discord_webhook_url)
                
                # 학습 데이터로 활용하기 위한 저장
                with open(os.path.join(cache_dir, f'realtime_data_{args.code}.json'), 'w') as f:
                    json.dump(real_time_info, f)
                
        except Exception as e:
            send_message(f"실시간 시세 정보 수집 실패: {e}", config.notification.discord_webhook_url)
        
        # 학습 파이프라인 실행
        model_id, results = trainer.run_training_pipeline(
            code=args.code,
            model_type=args.model_type,
            timesteps=args.timesteps,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        if model_id is not None:
            send_message(f"모델 학습 완료: {model_id}", config.notification.discord_webhook_url)
            if results is not None:
                send_message(f"테스트 결과: 수익률 {results['total_return']:.2f}%, 거래 횟수: {results['total_trades']}")
        else:
            send_message("모델 학습 실패", config.notification.discord_webhook_url)
            
    except Exception as e:
        send_message(f"학습 중 오류 발생: {e}", config.notification.discord_webhook_url)


def test_model(args):
    """
    모델 테스트
    
    Args:
        args: 명령행 인자
    """
    try:
        # 모델 ID 확인
        if args.model_id is None and not args.ensemble:
            # 가장 최근 모델 자동 선택
            models = list_available_models(args.output_dir)
            if not models:
                send_message("사용 가능한 모델이 없습니다.", config.notification.discord_webhook_url)
                return
            args.model_id = models[0]['model_id']
            send_message(f"최근 모델 자동 선택: {args.model_id}", config.notification.discord_webhook_url)
        
        # 데이터 가져오기
        fetcher = DataFetcher()
        data = fetcher.fetch_from_database(
            code=args.code, 
            start_date=args.start_date, 
            end_date=args.end_date
        )
        
        if data is None or len(data) == 0:
            send_message("테스트 데이터가 없습니다.", config.notification.discord_webhook_url)
            return
        
        # 데이터 전처리
        processor = RLDataManager()
        processed_data = processor.add_technical_indicators(data)
        processed_data = processor.normalize_data(processed_data)
        
        # 모델 로드
        if args.ensemble:
            # 앙상블 모델 로드
            ensemble_path = os.path.join(args.output_dir, "ensemble.pkl")
            model = ModelEnsemble.load_ensemble(ensemble_path)
            model_name = "ensemble"
        else:
            # 단일 모델 로드
            model = load_model_by_id(args.model_id, args.output_dir)
            model_name = args.model_id
        
        if model is None:
            send_message("모델을 로드할 수 없습니다.", config.notification.discord_webhook_url)
            return
        
        # 성능 시각화
        if not args.ensemble:
            visualize_model_performance(
                model_id=args.model_id, 
                test_data=processed_data, 
                models_dir=args.output_dir
            )
        else:
            # 앙상블 모델 테스트
            from korea_stock_auto.reinforcement.rl_models.rl_model import TradingEnvironment
            env = TradingEnvironment(df=processed_data)
            
            # 거래 시뮬레이션
            observation = env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(observation)
                observation, reward, done, info = env.step(action)
            
            # 결과 계산
            initial_balance = env.initial_balance
            final_value = env.balance
            if env.shares_held > 0:
                final_value += env.shares_held * processed_data.iloc[-1]['close']
                
            total_return = (final_value / initial_balance - 1) * 100
            
            # 거래 횟수 계산
            actions = [h['action'] for h in env.history]
            buy_count = actions.count(1)
            sell_count = actions.count(2)
            
            send_message(f"앙상블 모델 테스트 결과: 수익률 {total_return:.2f}%, 거래 횟수: {buy_count + sell_count}")
            
    except Exception as e:
        send_message(f"테스트 중 오류 발생: {e}", config.notification.discord_webhook_url)


def create_ensemble(args):
    """
    앙상블 모델 생성
    
    Args:
        args: 명령행 인자
    """
    try:
        # 트레이너 생성
        trainer = ModelTrainer(output_dir=args.output_dir)
        
        # 앙상블 생성
        ensemble = create_ensemble_from_best_models(trainer, top_n=args.top_n)
        
        if ensemble is None:
            send_message("앙상블 생성 실패", config.notification.discord_webhook_url)
        else:
            send_message(f"앙상블 생성 완료: {len(ensemble.models, config.notification.discord_webhook_url)}개 모델")
            
    except Exception as e:
        send_message(f"앙상블 생성 중 오류 발생: {e}", config.notification.discord_webhook_url)


def backtest(args):
    """
    백테스트 실행
    
    Args:
        args: 명령행 인자
    """
    try:
        # 모델 ID 확인
        if args.model_id is None and not args.ensemble:
            # 가장 최근 모델 자동 선택
            models = list_available_models(args.output_dir)
            if not models:
                send_message("사용 가능한 모델이 없습니다.", config.notification.discord_webhook_url)
                return
            args.model_id = models[0]['model_id']
            send_message(f"최근 모델 자동 선택: {args.model_id}", config.notification.discord_webhook_url)
        
        # 데이터 가져오기
        fetcher = DataFetcher()
        data = fetcher.fetch_from_database(
            code=args.code, 
            start_date=args.start_date, 
            end_date=args.end_date
        )
        
        if data is None or len(data) == 0:
            send_message("백테스트 데이터가 없습니다.", config.notification.discord_webhook_url)
            return
        
        # 데이터 전처리
        processor = RLDataManager()
        processed_data = processor.add_technical_indicators(data)
        processed_data = processor.normalize_data(processed_data)
        
        # 모델 로드
        if args.ensemble:
            # 앙상블 모델 로드
            ensemble_path = os.path.join(args.output_dir, "ensemble.pkl")
            model = ModelEnsemble.load_ensemble(ensemble_path)
            model_name = "ensemble"
        else:
            # 단일 모델 로드
            model = load_model_by_id(args.model_id, args.output_dir)
            model_name = args.model_id
        
        if model is None:
            send_message("모델을 로드할 수 없습니다.", config.notification.discord_webhook_url)
            return
        
        # 백테스트 환경 설정
        from korea_stock_auto.reinforcement.rl_models.rl_model import TradingEnvironment
        env = TradingEnvironment(
            df=processed_data,
            initial_balance=10000000,  # 1천만원
            commission=0.00015         # 0.015% 수수료
        )
        
        # 거래 시뮬레이션
        observation = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)
        
        # 백테스트 결과 계산
        history = pd.DataFrame(env.history)
        
        # 성능 지표 계산
        initial_balance = env.initial_balance
        final_balance = env.balance
        if env.shares_held > 0:
            final_balance += env.shares_held * processed_data.iloc[-1]['close']
        
        total_return = (final_balance / initial_balance - 1) * 100
        
        # 거래 통계
        buy_count = sum(1 for action in history['action'] if action == 1)
        sell_count = sum(1 for action in history['action'] if action == 2)
        
        # 수익성 지표
        winning_trades = 0
        losing_trades = 0
        profit_sum = 0
        loss_sum = 0
        
        for i in range(1, len(history)):
            if history.iloc[i]['action'] == 2:  # 매도 시점
                for j in range(i-1, -1, -1):
                    if history.iloc[j]['action'] == 1:  # 매수 시점 찾기
                        buy_price = history.iloc[j]['price']
                        sell_price = history.iloc[i]['price']
                        trade_return = (sell_price / buy_price - 1) * 100 - 0.03  # 왕복 수수료 0.03%
                        
                        if trade_return > 0:
                            winning_trades += 1
                            profit_sum += trade_return
                        else:
                            losing_trades += 1
                            loss_sum += abs(trade_return)
                        break
        
        total_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 백테스트 결과 출력
        send_message(f"백테스트 결과 ({model_name}, {args.code})")
        send_message(f"기간: {data.index[0].strftime('%Y-%m-%d', config.notification.discord_webhook_url)} ~ {data.index[-1].strftime('%Y-%m-%d')}")
        send_message(f"초기 자본: {initial_balance:,.0f}원")
        send_message(f"최종 자본: {final_balance:,.0f}원")
        send_message(f"총 수익률: {total_return:.2f}%", config.notification.discord_webhook_url)
        send_message(f"거래 횟수: {total_trades}회 (매수: {buy_count}회, 매도: {sell_count}회)")
        send_message(f"승률: {win_rate:.2f}% ({winning_trades}/{total_trades}, config.notification.discord_webhook_url)")
        
        # 연간 수익률 계산
        days = (data.index[-1] - data.index[0]).days
        annual_return = (total_return / days * 365) if days > 0 else 0
        send_message(f"연간 환산 수익률: {annual_return:.2f}%", config.notification.discord_webhook_url)
        
    except Exception as e:
        send_message(f"백테스트 중 오류 발생: {e}", config.notification.discord_webhook_url)


def compare_model_performance(args):
    """
    모델 성능 비교
    
    Args:
        args: 명령행 인자
    """
    try:
        # 비교할 모델 확인
        if args.models is None:
            # 모든 모델 자동 선택
            models_info = list_available_models(args.output_dir)
            if not models_info:
                send_message("사용 가능한 모델이 없습니다.", config.notification.discord_webhook_url)
                return
            args.models = [model['model_id'] for model in models_info]
        
        # 데이터 가져오기
        fetcher = DataFetcher()
        data = fetcher.fetch_from_database(
            code=args.code, 
            start_date=args.start_date, 
            end_date=args.end_date
        )
        
        if data is None or len(data) == 0:
            send_message("비교 데이터가 없습니다.", config.notification.discord_webhook_url)
            return
        
        # 데이터 전처리
        processor = RLDataManager()
        processed_data = processor.add_technical_indicators(data)
        processed_data = processor.normalize_data(processed_data)
        
        # 모델 비교
        output_path = os.path.join(args.output_dir, "model_comparison.csv")
        results = compare_models(
            model_ids=args.models, 
            test_data=processed_data, 
            models_dir=args.output_dir,
            output_path=output_path
        )
        
        if results.empty:
            send_message("모델 비교 결과가 없습니다.", config.notification.discord_webhook_url)
            return
        
        # 결과 출력
        send_message(f"모델 비교 결과 ({args.code}, config.notification.discord_webhook_url)")
        send_message(f"기간: {data.index[0].strftime('%Y-%m-%d', config.notification.discord_webhook_url)} ~ {data.index[-1].strftime('%Y-%m-%d')}")
        
        for _, row in results.iterrows():
            send_message(f"모델: {row['model_id']}, 수익률: {row['total_return']:.2f}%, 거래 횟수: {row['total_trades']}")
        
        send_message(f"비교 결과 저장 완료: {output_path}", config.notification.discord_webhook_url)
        
    except Exception as e:
        send_message(f"모델 비교 중 오류 발생: {e}", config.notification.discord_webhook_url)


def list_models(args):
    """
    사용 가능한 모델 목록 출력
    
    Args:
        args: 명령행 인자
    """
    try:
        # 모델 목록 가져오기
        models = list_available_models(args.output_dir)
        
        if not models:
            send_message("사용 가능한 모델이 없습니다.", config.notification.discord_webhook_url)
            return
        
        # 결과 출력
        send_message(f"사용 가능한 모델 목록 ({len(models, config.notification.discord_webhook_url)}개)")
        
        for model in models:
            created_at = model.get('created_at', 'N/A')
            model_type = model.get('model_type', 'N/A')
            total_return = model.get('total_return', 0)
            training_code = model.get('training_code', 'N/A')
            
            send_message(f"ID: {model['model_id']}, 유형: {model_type}, 학습 종목: {training_code}, 수익률: {total_return:.2f}%, 생성일: {created_at}")
        
    except Exception as e:
        send_message(f"모델 목록 조회 중 오류 발생: {e}", config.notification.discord_webhook_url)


def main():
    """메인 함수"""
    args = parse_args()
    
    send_message(f"강화학습 자동매매 시스템 - 모드: {args.mode}", config.notification.discord_webhook_url)
    
    # 명령에 따라 기능 실행
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'test':
        test_model(args)
    elif args.mode == 'ensemble':
        create_ensemble(args)
    elif args.mode == 'backtest':
        backtest(args)
    elif args.mode == 'compare':
        compare_model_performance(args)
    elif args.mode == 'list':
        list_models(args)
    else:
        send_message(f"알 수 없는 모드: {args.mode}", config.notification.discord_webhook_url)


if __name__ == "__main__":
    main() 