import pandas as pd
import os

EXCEL_PATH = os.path.join(os.path.dirname(__file__), '../../../..', '한국투자증권api문서/국내증시/종목정보/상품기본조회[v1_국내주식-029].xlsx')

def load_basic_info():
    """엑셀 파일에서 종목 기본정보 데이터프레임을 반환"""
    return pd.read_excel(EXCEL_PATH)

def get_basic_info_by_code(code: str):
    """종목코드로 기본정보를 반환"""
    df = load_basic_info()
    row = df[df['종목코드'] == code]
    if row.empty:
        return None
    return row.to_dict(orient='records')[0] 