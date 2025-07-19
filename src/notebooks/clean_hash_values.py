import pandas as pd
import numpy as np

def clean_hash_values(df, columns=None):
    """
    데이터프레임에서 '#'이 들어간 셀을 None 값으로 변경하는 함수
    
    Parameters:
    df: pandas DataFrame
    columns: 처리할 컬럼 리스트 (None이면 모든 컬럼 처리)
    
    Returns:
    cleaned_df: 정리된 데이터프레임
    """
    # 데이터프레임 복사
    cleaned_df = df.copy()
    
    # 처리할 컬럼 결정
    if columns is None:
        columns = cleaned_df.columns.tolist()
    
    # 각 컬럼에서 '#' 값 찾아서 None으로 변경
    for col in columns:
        if col in cleaned_df.columns:
            # 문자열 컬럼인 경우에만 처리
            if cleaned_df[col].dtype == 'object':
                # '#'이 포함된 모든 값들을 None으로 변경
                cleaned_df[col] = cleaned_df[col].replace('#', None)
                cleaned_df[col] = cleaned_df[col].replace('#', np.nan)
                
                # '#'으로만 구성된 값들도 None으로 변경
                cleaned_df[col] = cleaned_df[col].apply(
                    lambda x: None if isinstance(x, str) and x.strip() == '#' else x
                )
    
    return cleaned_df

def clean_hash_values_detailed(df, columns=None, report=True):
    """
    데이터프레임에서 '#'이 들어간 셀을 None 값으로 변경하고 상세한 보고서 제공
    
    Parameters:
    df: pandas DataFrame
    columns: 처리할 컬럼 리스트 (None이면 모든 컬럼 처리)
    report: 보고서 출력 여부
    
    Returns:
    cleaned_df: 정리된 데이터프레임
    report_dict: 정리 결과 보고서
    """
    # 데이터프레임 복사
    cleaned_df = df.copy()
    
    # 처리할 컬럼 결정
    if columns is None:
        columns = cleaned_df.columns.tolist()
    
    # 보고서용 딕셔너리
    report_dict = {
        'total_rows': len(df),
        'processed_columns': [],
        'hash_counts': {},
        'total_hash_found': 0
    }
    
    # 각 컬럼에서 '#' 값 찾아서 None으로 변경
    for col in columns:
        if col in cleaned_df.columns:
            # 문자열 컬럼인 경우에만 처리
            if cleaned_df[col].dtype == 'object':
                # 변경 전 '#' 개수 확인
                hash_count = cleaned_df[col].astype(str).str.contains('#', na=False).sum()
                
                if hash_count > 0:
                    # '#'이 포함된 모든 값들을 None으로 변경
                    cleaned_df[col] = cleaned_df[col].replace('#', None)
                    cleaned_df[col] = cleaned_df[col].replace('#', np.nan)
                    
                    # '#'으로만 구성된 값들도 None으로 변경
                    cleaned_df[col] = cleaned_df[col].apply(
                        lambda x: None if isinstance(x, str) and x.strip() == '#' else x
                    )
                    
                    # 보고서에 추가
                    report_dict['processed_columns'].append(col)
                    report_dict['hash_counts'][col] = hash_count
                    report_dict['total_hash_found'] += hash_count
    
    # 보고서 출력
    if report:
        print("=" * 60)
        print("'#' 값 정리 결과 보고서")
        print("=" * 60)
        print(f"총 행 수: {report_dict['total_rows']}")
        print(f"처리된 컬럼 수: {len(report_dict['processed_columns'])}")
        print(f"총 발견된 '#' 개수: {report_dict['total_hash_found']}")
        
        if report_dict['processed_columns']:
            print(f"\n처리된 컬럼별 '#' 개수:")
            for col in report_dict['processed_columns']:
                print(f"  {col}: {report_dict['hash_counts'][col]}개")
        else:
            print("\n처리된 컬럼이 없습니다.")
        
        print("=" * 60)
    
    return cleaned_df, report_dict

def clean_hash_values_advanced(df, columns=None, replacement_value=None, report=True):
    """
    고급 '#' 값 정리 함수 - 다양한 옵션 제공
    
    Parameters:
    df: pandas DataFrame
    columns: 처리할 컬럼 리스트 (None이면 모든 컬럼 처리)
    replacement_value: '#'을 대체할 값 (None이면 np.nan 사용)
    report: 보고서 출력 여부
    
    Returns:
    cleaned_df: 정리된 데이터프레임
    """
    # 데이터프레임 복사
    cleaned_df = df.copy()
    
    # 대체값 설정
    if replacement_value is None:
        replacement_value = np.nan
    
    # 처리할 컬럼 결정
    if columns is None:
        columns = cleaned_df.columns.tolist()
    
    # 각 컬럼에서 '#' 값 찾아서 변경
    for col in columns:
        if col in cleaned_df.columns:
            # 문자열 컬럼인 경우에만 처리
            if cleaned_df[col].dtype == 'object':
                # '#'이 포함된 모든 값들을 대체값으로 변경
                cleaned_df[col] = cleaned_df[col].replace('#', replacement_value)
                
                # '#'으로만 구성된 값들도 대체값으로 변경
                cleaned_df[col] = cleaned_df[col].apply(
                    lambda x: replacement_value if isinstance(x, str) and x.strip() == '#' else x
                )
    
    if report:
        print(f"'#' 값을 {replacement_value}로 변경 완료!")
    
    return cleaned_df

# 사용 예시
if __name__ == "__main__":
    # 예시 데이터 생성
    example_data = {
        'name': ['John', '#', 'Jane', 'Bob', '#'],
        'age': [25, '#', 30, 35, 40],
        'city': ['NYC', 'LA', '#', 'Chicago', 'Boston'],
        'score': [85, 90, '#', 95, 88]
    }
    
    df = pd.DataFrame(example_data)
    
    print("원본 데이터:")
    print(df)
    print("\n")
    
    # 기본 정리
    cleaned_df = clean_hash_values(df)
    print("기본 정리 후:")
    print(cleaned_df)
    print("\n")
    
    # 상세 보고서와 함께 정리
    cleaned_df2, report = clean_hash_values_detailed(df)
    
    # 특정 컬럼만 정리
    cleaned_df3 = clean_hash_values(df, columns=['name', 'city'])
    print("특정 컬럼만 정리 후:")
    print(cleaned_df3)
    print("\n")
    
    # 다른 값으로 대체
    cleaned_df4 = clean_hash_values_advanced(df, replacement_value='MISSING')
    print("'MISSING'으로 대체 후:")
    print(cleaned_df4) 