import pandas as pd
import os
import glob
import logging

logger = logging.getLogger("stock_auto_excel_reader")
logging.basicConfig(level=logging.INFO)

# 행과 열 표시 설정
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', 500)

def get_api_spec_from_excel(file_path: str) -> dict:
    """
    Excel 파일에서 API 명세 정보를 읽어옵니다.
    
    Args:
        file_path (str): Excel 파일 경로
        
    Returns:
        dict: API 명세 정보를 담은 딕셔너리
    """
    api_specs = {}
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"Excel 파일을 찾을 수 없습니다: {file_path}")
            return api_specs
            
        logger.info(f"Excel 파일 로드 시작: {file_path}")
        
        # Excel 파일의 모든 시트 확인
        xls = pd.ExcelFile(file_path)
        logger.info(f"사용 가능한 시트: {xls.sheet_names}")
        
        # 각 시트를 확인하여 API 정보가 있는 시트 찾기
        for sheet_name in xls.sheet_names:
            logger.info(f"시트 '{sheet_name}' 분석 중...")
            
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                logger.info(f"시트 '{sheet_name}' 로드 완료. 크기: {df.shape}")
                
                # DataFrame을 문자열로 변환하여 전체 내용 검색
                df_str = df.astype(str).to_string()
                
                # API 관련 키워드 확인
                api_keywords = ['TR_ID', 'API', '요청', '응답', 'URL', 'Path', 'Parameter', '파라미터']
                has_api_info = any(keyword in df_str for keyword in api_keywords)
                
                if has_api_info:
                    logger.info(f"시트 '{sheet_name}'에서 API 정보 발견")
                    
                    # API 정보 추출
                    api_info = extract_api_info_from_dataframe(df, sheet_name)
                    if api_info:
                        api_specs[sheet_name] = api_info
                        logger.info(f"시트 '{sheet_name}'에서 API 정보 추출 완료")
                
            except Exception as e:
                logger.warning(f"시트 '{sheet_name}' 처리 중 오류: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Excel 파일 처리 중 오류 발생: {e}", exc_info=True)
        
    return api_specs

def extract_api_info_from_dataframe(df: pd.DataFrame, sheet_name: str) -> dict:
    """
    DataFrame에서 API 정보를 추출합니다.
    
    Args:
        df: pandas DataFrame
        sheet_name: 시트 이름
        
    Returns:
        dict: 추출된 API 정보
    """
    api_info = {
        'sheet_name': sheet_name,
        'tr_id': '',
        'api_path': '',
        'method': 'POST',
        'request_params': [],
        'response_fields': [],
        'description': '',
        'domain_prod': '',
        'domain_test': ''
    }
    
    # DataFrame을 문자열로 변환
    df = df.astype(str)
    
    # 디버그: 전체 내용 출력 (처음 20행)
    print(f"\n=== 시트 '{sheet_name}' 전체 내용 분석 ===")
    for i in range(min(25, len(df))):
        row_values = [str(val) for val in df.iloc[i].values if str(val) != 'nan']
        if row_values:
            print(f"행 {i:2d}: {' | '.join(row_values)}")
    
    # 기본 정보 추출
    for i in range(len(df)):
        for j in range(len(df.columns)):
            cell_value = str(df.iloc[i, j]).strip()
            next_cell = str(df.iloc[i, j + 1]).strip() if j + 1 < len(df.columns) else ''
            
            # TR_ID 찾기
            if 'TR_ID' in cell_value or 'TR-ID' in cell_value:
                if next_cell != 'nan' and len(next_cell) > 5:
                    api_info['tr_id'] = next_cell
                    print(f"✓ TR_ID 발견: {next_cell}")
            
            # HTTP Method 찾기
            if 'HTTP Method' in cell_value or 'Method' in cell_value:
                if next_cell != 'nan':
                    api_info['method'] = next_cell
                    print(f"✓ HTTP Method 발견: {next_cell}")
            
            # Domain 정보 찾기
            if '실전 Domain' in cell_value or 'Production' in cell_value:
                if next_cell != 'nan' and 'http' in next_cell:
                    api_info['domain_prod'] = next_cell
                    print(f"✓ 실전 Domain 발견: {next_cell}")
            
            if '모의 Domain' in cell_value or 'Test' in cell_value:
                if next_cell != 'nan' and 'http' in next_cell:
                    api_info['domain_test'] = next_cell
                    print(f"✓ 모의 Domain 발견: {next_cell}")
            
            # API 명 또는 설명 찾기
            if 'API 명' in cell_value or 'API명' in cell_value:
                if next_cell != 'nan':
                    api_info['description'] = next_cell
                    print(f"✓ API 명 발견: {next_cell}")

    # URL Path 찾기 - 더 상세한 검색
    print(f"\n--- URL Path 검색 ---")
    for i in range(len(df)):
        row_str = ' '.join([str(df.iloc[i, j]) for j in range(len(df.columns)) if str(df.iloc[i, j]) != 'nan'])
        
        # URL 패턴 찾기
        if '/uapi/' in row_str or '/domestic-stock/' in row_str or 'quotations' in row_str:
            print(f"행 {i}: URL 패턴 발견 - {row_str}")
            # URL 경로 추출
            for part in row_str.split():
                if '/uapi/' in part:
                    api_info['api_path'] = part
                    print(f"✓ API Path 발견: {part}")
                    break

    # Request Parameter 섹션 찾기
    print(f"\n--- Request Parameter 검색 ---")
    param_section_found = False
    
    for i in range(len(df)):
        row_str = ' '.join([str(df.iloc[i, j]) for j in range(len(df.columns)) if str(df.iloc[i, j]) != 'nan'])
        
        # 파라미터 섹션 헤더 찾기 (영어 키워드 우선)
        if ('Request Parameter' in row_str or 'Request Header' in row_str or 
            'Query Parameter' in row_str or 'Path Parameter' in row_str or
            ('Request' in row_str and 'Parameter' in row_str)):
            print(f"행 {i}: Request Parameter 섹션 시작")
            param_section_found = True
            
            # 파라미터 테이블 파싱 (다음 행부터)
            for pi in range(i + 1, min(i + 25, len(df))):
                param_name = str(df.iloc[pi, 0]).strip()
                
                # 유효한 파라미터 이름인지 확인 (영어 키워드 제외)
                if (param_name != 'nan' and len(param_name) > 0 and 
                    not any(skip in param_name.lower() for skip in [
                        '항목명', '항목', '구분', '타입', 'type', '설명', 'description',
                        'element', 'required', 'length', 'response', 'output'
                    ])):
                    
                    param_info = {'name': param_name}
                    
                    # 나머지 열에서 정보 추출
                    for col in range(1, len(df.columns)):
                        cell_val = str(df.iloc[pi, col]).strip()
                        if cell_val != 'nan' and len(cell_val) > 0:
                            if col == 1:  # 보통 두 번째 열이 설명
                                param_info['description'] = cell_val
                            elif col == 2:  # 세 번째 열이 타입인 경우
                                param_info['type'] = cell_val
                            elif cell_val in ['Y', 'N', '필수', '선택', 'Required', 'Optional']:
                                param_info['required'] = cell_val in ['Y', '필수', 'Required']
                    
                    api_info['request_params'].append(param_info)
                    print(f"  - 파라미터: {param_name} ({param_info.get('description', '')}) [{param_info.get('type', '')}]")
                
                # 다음 섹션이 시작되면 중단 (영어 키워드 추가)
                next_row_str = ' '.join([str(df.iloc[pi, j]) for j in range(len(df.columns)) if str(df.iloc[pi, j]) != 'nan'])
                if ('Response' in next_row_str or 'Output' in next_row_str or 
                    'Example' in next_row_str or '예제' in next_row_str):
                    break
            break

    # Response Fields 섹션 찾기
    print(f"\n--- Response Fields 검색 ---")
    
    for i in range(len(df)):
        row_str = ' '.join([str(df.iloc[i, j]) for j in range(len(df.columns)) if str(df.iloc[i, j]) != 'nan'])
        
        # 응답 섹션 헤더 찾기 (영어 키워드 우선)
        if ('Response Parameter' in row_str or 'Response Header' in row_str or 
            'Response Field' in row_str or 'Output' in row_str or
            ('Response' in row_str and ('Parameter' in row_str or 'Field' in row_str))):
            print(f"행 {i}: Response Fields 섹션 시작")
            
            # 응답 필드 테이블 파싱
            for ri in range(i + 1, min(i + 40, len(df))):
                field_name = str(df.iloc[ri, 0]).strip()
                
                # 유효한 필드 이름인지 확인 (영어 키워드 제외)
                if (field_name != 'nan' and len(field_name) > 0 and 
                    not any(skip in field_name.lower() for skip in [
                        '항목명', '항목', '구분', 'example', '예제', 'element', 
                        'required', 'length', 'type', 'description'
                    ])):
                    
                    field_info = {'name': field_name}
                    
                    # 나머지 열에서 정보 추출
                    for col in range(1, len(df.columns)):
                        cell_val = str(df.iloc[ri, col]).strip()
                        if cell_val != 'nan' and len(cell_val) > 0:
                            if col == 1:  # 보통 두 번째 열이 설명
                                field_info['description'] = cell_val
                            elif col == 2:  # 세 번째 열이 타입인 경우
                                field_info['type'] = cell_val
                    
                    api_info['response_fields'].append(field_info)
                    print(f"  - 응답필드: {field_name} ({field_info.get('description', '')}) [{field_info.get('type', '')}]")
                
                # 예제나 다른 섹션이 시작되면 중단 (영어 키워드 추가)
                next_row_str = ' '.join([str(df.iloc[ri, j]) for j in range(len(df.columns)) if str(df.iloc[ri, j]) != 'nan'])
                if ('Example' in next_row_str or '예제' in next_row_str or 
                    'Request' in next_row_str):
                    break
            break

    return api_info

def print_api_specs(api_specs: dict):
    """추출된 API 명세 정보를 출력합니다."""
    for sheet_name, api_info in api_specs.items():
        print(f"\n=== 시트: {sheet_name} ===")
        print(f"TR_ID: {api_info.get('tr_id', '정보 없음')}")
        print(f"API 경로: {api_info.get('api_path', '정보 없음')}")
        print(f"HTTP 메소드: {api_info.get('method', 'POST')}")
        print(f"설명: {api_info.get('description', '정보 없음')}")
        
        print(f"\n요청 파라미터 ({len(api_info.get('request_params', []))}개):")
        for param in api_info.get('request_params', []):
            required_str = " (필수)" if param.get('required') else ""
            desc_str = f" - {param.get('description', '')}" if param.get('description') else ""
            print(f"  - {param['name']}{required_str}{desc_str}")
        
        print(f"\n응답 필드 ({len(api_info.get('response_fields', []))}개):")
        for field in api_info.get('response_fields', []):
            desc_str = f" - {field.get('description', '')}" if field.get('description') else ""
            print(f"  - {field['name']}{desc_str}")

if __name__ == '__main__':
    # 사용자가 지정한 Excel 파일 경로
    excel_file_path = r"C:\Users\piome\Desktop\백엔드 부트캠프\koreainvestment-autotrade-main\주식현재가 일자별[v1_국내주식-010].xlsx"
    
    if not os.path.exists(excel_file_path):
        logger.error(f"지정된 Excel 파일이 존재하지 않습니다: {excel_file_path}")
        print("파일 경로를 확인해주세요.")
    else:
        logger.info(f"Excel 파일 분석 시작: {excel_file_path}")
        
        # API 명세 추출
        api_documentation = get_api_spec_from_excel(excel_file_path)
        
        if api_documentation:
            print(f"\n총 {len(api_documentation)}개의 API 명세를 발견했습니다.")
            print_api_specs(api_documentation)
            
            # 일자별 API 정보 반환 (historical_price.py 수정용)
            for sheet_name, api_info in api_documentation.items():
                if '일자별' in sheet_name or 'daily' in sheet_name.lower() or api_info.get('tr_id'):
                    print(f"\n=== 일자별 API 정보 (historical_price.py 수정용) ===")
                    print(f"TR_ID: {api_info.get('tr_id')}")
                    print(f"API_PATH: {api_info.get('api_path')}")
                    print("주요 요청 파라미터:")
                    for param in api_info.get('request_params', []):
                        print(f"  - {param['name']}")
                    print("주요 응답 필드:")
                    for field in api_info.get('response_fields', []):
                        print(f"  - {field['name']}")
                    break
        else:
            print("API 명세 정보를 찾지 못했습니다. Excel 파일 내용을 확인해주세요.") 