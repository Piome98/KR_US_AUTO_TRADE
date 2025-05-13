"""
한국 주식 자동매매 - 강화학습 모델 평가 모듈

훈련된 강화학습 모델에 대한 평가 및 분석 기능 제공
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from korea_stock_auto.reinforcement.rl_models import TradingEnvironment, RLModel
from korea_stock_auto.utils import send_message

logger = logging.getLogger("stock_auto")

class ModelEvaluator:
    """강화학습 모델 평가 클래스"""
    
    def __init__(self, output_dir="models"):
        """
        평가자 초기화
        
        Args:
            output_dir (str): 출력 디렉토리 경로
        """
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)

    def evaluate_model(self, model: RLModel, test_data: pd.DataFrame, output_prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        모델 평가 수행
        
        Args:
            model (RLModel): 평가할 모델
            test_data (pd.DataFrame): 테스트 데이터
            output_prefix (str): 결과 파일 저장 시 접두사
            
        Returns:
            dict: 평가 결과
        """
        if model is None or test_data is None or test_data.empty:
            logger.error("모델 또는 테스트 데이터가 없습니다.")
            return {}
            
        try:
            # 테스트 환경 설정
            env = TradingEnvironment(df=test_data)
            observation = env.reset()
            
            # 평가를 위한 변수 초기화
            done = False
            total_reward = 0
            history = []
            
            # 에피소드 진행
            while not done:
                # 모델 예측
                action, _ = model.predict(observation)
                
                # 환경에서 액션 실행
                observation, reward, done, info = env.step(action)
                
                # 이력 및 보상 누적
                total_reward += reward
                history.append({
                    'step': len(history),
                    'action': action,
                    'reward': reward,
                    'portfolio_value': info['portfolio_value'],
                    'price': info['price'],
                    'shares_held': info['shares_held']
                })
            
            # 이력 데이터프레임 변환
            history_df = pd.DataFrame(history)
            
            # 성능 지표 계산
            performance_metrics = self._calculate_performance_metrics(history_df, env.initial_balance)
            
            # 결과 저장 및 시각화
            if output_prefix:
                # 이력 저장
                history_df.to_csv(os.path.join(self.output_dir, f"{output_prefix}_test_history.csv"), index=False)
                
                # 성능 시각화
                self._save_performance_plots(history_df, output_prefix)
            
            # 결과 로깅
            logger.info(f"모델 평가 완료: 총 보상 {total_reward:.2f}, "
                        f"최종 수익률 {performance_metrics['total_return']:.2%}")
            
            return {
                'total_reward': float(total_reward),
                'metrics': performance_metrics,
                'history': history
            }
            
        except Exception as e:
            logger.error(f"모델 평가 실패: {e}", exc_info=True)
            return {}
    
    def _calculate_performance_metrics(self, history_df: pd.DataFrame, initial_balance: float) -> Dict[str, float]:
        """
        성능 지표 계산
        
        Args:
            history_df (pd.DataFrame): 거래 이력 데이터프레임
            initial_balance (float): 초기 투자금
            
        Returns:
            dict: 성능 지표
        """
        try:
            if history_df.empty:
                return {}
                
            # 기본 지표
            final_portfolio_value = history_df['portfolio_value'].iloc[-1]
            total_return = (final_portfolio_value / initial_balance) - 1
            
            # 일별 수익률 계산 (포트폴리오 가치 변화율)
            history_df['daily_return'] = history_df['portfolio_value'].pct_change()
            history_df.loc[0, 'daily_return'] = 0  # 첫날은 변화 없음
            
            # 수익 거래와 손실 거래 구분
            history_df['is_profitable'] = history_df['daily_return'] > 0
            
            # 거래 횟수 (액션이 변경된 시점)
            history_df['action_changed'] = history_df['action'].diff() != 0
            total_trades = history_df['action_changed'].sum()
            
            # 최대 낙폭 (MDD) 계산
            history_df['cummax'] = history_df['portfolio_value'].cummax()
            history_df['drawdown'] = (history_df['portfolio_value'] / history_df['cummax']) - 1
            max_drawdown = history_df['drawdown'].min()
            
            # 변동성 (일별 수익률의 표준편차)
            volatility = history_df['daily_return'].std()
            
            # 샤프 비율 (연간화, 무위험 수익률 가정 0%)
            # 252는 일년 중 거래일 수
            sharpe_ratio = 0
            if volatility > 0:
                avg_daily_return = history_df['daily_return'].mean()
                sharpe_ratio = (avg_daily_return / volatility) * np.sqrt(252)
            
            # 승률 계산
            if total_trades > 0:
                profitable_trades = history_df[history_df['action_changed'] & history_df['is_profitable']].shape[0]
                win_rate = profitable_trades / total_trades
            else:
                win_rate = 0
                
            # 최대 연속 손실 횟수
            consecutive_losses = 0
            max_consecutive_losses = 0
            
            for is_profit in history_df['is_profitable']:
                if not is_profit:
                    consecutive_losses += 1
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                else:
                    consecutive_losses = 0
            
            return {
                'total_return': float(total_return),
                'max_drawdown': float(max_drawdown),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'win_rate': float(win_rate),
                'total_trades': int(total_trades),
                'max_consecutive_losses': int(max_consecutive_losses),
                'final_value': float(final_portfolio_value)
            }
            
        except Exception as e:
            logger.error(f"성능 지표 계산 실패: {e}", exc_info=True)
            return {}
    
    def _save_performance_plots(self, history_df: pd.DataFrame, prefix: str) -> None:
        """
        성능 시각화 및 저장
        
        Args:
            history_df (pd.DataFrame): 거래 이력 데이터프레임
            prefix (str): 파일명 접두사
        """
        try:
            if history_df.empty:
                return
                
            # 1. 포트폴리오 가치 변화 그래프
            plt.figure(figsize=(12, 6))
            plt.plot(history_df['step'], history_df['portfolio_value'])
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Step')
            plt.ylabel('Portfolio Value (KRW)')
            plt.grid(True)
            plt.savefig(os.path.join(self.figures_dir, f"{prefix}_portfolio_value.png"))
            plt.close()
            
            # 2. 가격과 포트폴리오 가치 비교 그래프
            plt.figure(figsize=(12, 6))
            
            # 두 축 설정 (가격과 포트폴리오 가치를 같은 그래프에 표시)
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            
            # 가격 그래프
            ax1.plot(history_df['step'], history_df['price'], 'b-', label='Stock Price')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Stock Price (KRW)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # 포트폴리오 가치 그래프
            ax2.plot(history_df['step'], history_df['portfolio_value'], 'r-', label='Portfolio Value')
            ax2.set_ylabel('Portfolio Value (KRW)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            plt.title('Stock Price vs Portfolio Value')
            plt.grid(True)
            
            # 범례 추가
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.savefig(os.path.join(self.figures_dir, f"{prefix}_price_vs_portfolio.png"))
            plt.close()
            
            # 3. 액션 분포 그래프
            plt.figure(figsize=(10, 6))
            action_counts = history_df['action'].value_counts()
            action_labels = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
            
            # 액션 이름으로 변환
            action_counts.index = [action_labels.get(a, f'Action {a}') for a in action_counts.index]
            
            action_counts.plot(kind='bar', color=['gray', 'green', 'red'])
            plt.title('Action Distribution')
            plt.xlabel('Action')
            plt.ylabel('Count')
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, f"{prefix}_action_distribution.png"))
            plt.close()
            
            # 4. 각 스텝에서의 액션 시각화
            plt.figure(figsize=(14, 6))
            
            # 가격 그래프
            plt.plot(history_df['step'], history_df['price'], 'k-', label='Price')
            
            # 액션별 색상
            colors = {0: 'gray', 1: 'green', 2: 'red'}
            action_labels = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
            
            # 액션별 마커 표시
            for action in [0, 1, 2]:
                mask = history_df['action'] == action
                if mask.any():
                    plt.scatter(
                        history_df.loc[mask, 'step'],
                        history_df.loc[mask, 'price'],
                        color=colors[action],
                        label=action_labels[action],
                        alpha=0.7,
                        s=50
                    )
            
            plt.title('Trading Actions Over Time')
            plt.xlabel('Step')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.figures_dir, f"{prefix}_trading_actions.png"))
            plt.close()
            
            logger.info(f"성능 시각화 저장 완료: {prefix}")
            
        except Exception as e:
            logger.error(f"성능 시각화 저장 실패: {e}", exc_info=True)

    def compare_models(self, model_results: List[Dict[str, Any]], model_names: List[str]) -> None:
        """
        여러 모델 성능 비교
        
        Args:
            model_results (list): 각 모델의 평가 결과 목록
            model_names (list): 모델 이름 목록
        """
        try:
            if not model_results or len(model_results) != len(model_names):
                logger.error("모델 결과와 이름 목록이 일치하지 않습니다.")
                return
                
            # 비교를 위한 데이터프레임 생성
            comparison_data = []
            
            for i, result in enumerate(model_results):
                if 'metrics' not in result:
                    continue
                    
                metrics = result['metrics']
                metrics['model_name'] = model_names[i]
                comparison_data.append(metrics)
            
            if not comparison_data:
                logger.error("비교할 모델 결과가 없습니다.")
                return
                
            # 데이터프레임 변환
            comparison_df = pd.DataFrame(comparison_data)
            
            # 결과 저장
            comparison_df.to_csv(os.path.join(self.output_dir, "model_comparison.csv"), index=False)
            
            # 주요 지표 시각화
            metrics_to_plot = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            
            for metric in metrics_to_plot:
                if metric in comparison_df.columns:
                    plt.figure(figsize=(10, 6))
                    
                    # 지표에 따라 다른 색상 사용 (긍정적인 지표는 초록색, 부정적인 지표는 빨간색)
                    colors = ['green' if m in ['total_return', 'sharpe_ratio', 'win_rate'] else 'red' 
                              for m in [metric] * len(comparison_df)]
                    
                    comparison_df.plot(
                        x='model_name', 
                        y=metric, 
                        kind='bar', 
                        color=colors,
                        title=f'Model Comparison - {metric.replace("_", " ").title()}',
                        grid=True
                    )
                    
                    plt.ylabel(metric.replace('_', ' ').title())
                    plt.xlabel('Model')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.figures_dir, f"comparison_{metric}.png"))
                    plt.close()
            
            logger.info(f"모델 비교 시각화 저장 완료")
            
        except Exception as e:
            logger.error(f"모델 비교 실패: {e}", exc_info=True) 