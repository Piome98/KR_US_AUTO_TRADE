import pandas as pd
import os

EXCEL_PATH = os.path.join(os.path.dirname(__file__), '../../../..', '한국투자증권api문서/국내증시/종목정보/국내주식 재무비율[v1_국내주식-080].xlsx')

def load_financial_ratios():
    """엑셀 파일에서 재무비율 데이터프레임을 반환"""
    return pd.read_excel(EXCEL_PATH)

def get_financial_ratios_by_code(code: str):
    """종목코드로 재무비율 정보를 반환"""
    df = load_financial_ratios()
    row = df[df['종목코드'] == code]
    if row.empty:
        return None
    return row.to_dict(orient='records')[0] 