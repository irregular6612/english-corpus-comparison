import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def remove_outliers(df, column, method='iqr'):
    """
    아웃라이어를 제거하는 함수
    
    Parameters:
    df: DataFrame
    column: 아웃라이어를 제거할 컬럼명
    method: 'iqr' (기본값) 또는 'zscore'
    
    Returns:
    아웃라이어가 제거된 DataFrame
    """
    if method == 'iqr':
        # IQR 방법으로 아웃라이어 제거
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 아웃라이어가 아닌 데이터만 필터링
        filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        
    elif method == 'zscore':
        # Z-score 방법으로 아웃라이어 제거 (|z-score| > 3)
        z_scores = np.abs(stats.zscore(df[column], nan_policy='omit'))
        filtered_df = df[z_scores < 3]
    
    return filtered_df

def plot_with_outlier_removal(df0: pd.DataFrame, df1: pd.DataFrame, df2: pd.DataFrame, 
                             outlier_method='iqr', show_stats=True):
    """
    아웃라이어를 제거하고 박스플롯을 그리는 함수
    
    Parameters:
    df0, df1, df2: 입력 DataFrame들
    outlier_method: 아웃라이어 제거 방법 ('iqr' 또는 'zscore')
    show_stats: 통계 정보 출력 여부
    """
    # 데이터 병합
    sample_df = pd.merge(df0, df1, how='inner', on='Word')[['Word', 'SUBTLRWF', 'CSAT_RFreq', 'RFreq_HAL']]
    sample_df['Freq_type'] = 'HF'
    sample2_df = pd.merge(df0, df2, how='inner', on='Word')[['Word', 'SUBTLRWF', 'CSAT_RFreq', 'RFreq_HAL']]
    sample2_df['Freq_type'] = 'LF'
    sample_df = pd.concat([sample_df, sample2_df], ignore_index=True)
    
    target_col = ['SUBTLRWF', 'CSAT_RFreq', 'RFreq_HAL']
    
    # 아웃라이어 제거 전 통계
    if show_stats:
        print("=== 아웃라이어 제거 전 통계 ===")
        print(sample_df[target_col].describe())
        print(f"총 데이터 수: {len(sample_df)}")
    
    # 각 Corpus별로 아웃라이어 제거
    cleaned_dfs = []
    for freq_type in ['HF', 'LF']:
        freq_data = sample_df[sample_df['Freq_type'] == freq_type].copy()
        
        for col in target_col:
            # 각 컬럼별로 아웃라이어 제거
            cleaned_data = remove_outliers(freq_data, col, method=outlier_method)
            freq_data = cleaned_data
        
        cleaned_dfs.append(freq_data)
    
    # 아웃라이어가 제거된 데이터 병합
    cleaned_sample_df = pd.concat(cleaned_dfs, ignore_index=True)
    
    # 아웃라이어 제거 후 통계
    if show_stats:
        print("\n=== 아웃라이어 제거 후 통계 ===")
        print(cleaned_sample_df[target_col].describe())
        print(f"아웃라이어 제거 후 데이터 수: {len(cleaned_sample_df)}")
        print(f"제거된 데이터 수: {len(sample_df) - len(cleaned_sample_df)}")
        print(f"제거 비율: {((len(sample_df) - len(cleaned_sample_df)) / len(sample_df) * 100):.2f}%")
    
    # 데이터 형태 변환 (melt)
    melted_df = cleaned_sample_df.melt(id_vars=['Word', 'Freq_type'], 
                                      var_name='Corpus', value_name='Freq')
    
    # 박스플롯 생성
    plt.figure(figsize=(8, 10))
    
    # 서브플롯 1: 아웃라이어 제거 전
    plt.subplot(2, 1, 1)
    original_melted = sample_df.melt(id_vars=['Word', 'Freq_type'], 
                                   var_name='Corpus', value_name='Freq')
    sns.boxplot(data=original_melted, x='Freq_type', y='Freq', hue='Corpus')
    plt.title('아웃라이어 제거 전 박스플롯')
    plt.ylabel('Frequency')
    
    # 서브플롯 2: 아웃라이어 제거 후
    plt.subplot(2, 1, 2)
    sns.boxplot(data=melted_df, x='Freq_type', y='Freq', hue='Corpus')
    plt.title(f'아웃라이어 제거 후 박스플롯 ({outlier_method.upper()} 방법)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return cleaned_sample_df, melted_df

def plot_simple_outlier_removal(df0: pd.DataFrame, df1: pd.DataFrame, df2: pd.DataFrame, 
                               outlier_method='iqr'):
    """
    간단한 버전: 아웃라이어 제거 후 박스플롯만 그리기
    """
    # 데이터 병합
    sample_df = pd.merge(df0, df1, how='inner', on='Word')[['Word', 'SUBTLRWF', 'CSAT_RFreq', 'RFreq_HAL']]
    sample_df['Freq_type'] = 'HF'
    sample2_df = pd.merge(df0, df2, how='inner', on='Word')[['Word', 'SUBTLRWF', 'CSAT_RFreq', 'RFreq_HAL']]
    sample2_df['Freq_type'] = 'LF'
    sample_df = pd.concat([sample_df, sample2_df], ignore_index=True)
    
    target_col = ['SUBTLRWF', 'CSAT_RFreq', 'RFreq_HAL']
    
    # 각 Corpus별로 아웃라이어 제거
    cleaned_dfs = []
    for freq_type in ['HF', 'LF']:
        freq_data = sample_df[sample_df['Freq_type'] == freq_type].copy()
        
        for col in target_col:
            cleaned_data = remove_outliers(freq_data, col, method=outlier_method)
            freq_data = cleaned_data
        
        cleaned_dfs.append(freq_data)
    
    # 아웃라이어가 제거된 데이터 병합
    cleaned_sample_df = pd.concat(cleaned_dfs, ignore_index=True)
    
    # 데이터 형태 변환
    melted_df = cleaned_sample_df.melt(id_vars=['Word', 'Freq_type'], 
                                      var_name='Corpus', value_name='Freq')
    
    # 박스플롯 생성
    plt.figure(figsize=(6, 8))
    sns.boxplot(data=melted_df, x='Freq_type', y='Freq', hue='Corpus')
    plt.title(f'아웃라이어 제거 후 박스플롯 ({outlier_method.upper()} 방법)')
    plt.ylabel('Frequency')
    plt.show()
    
    return cleaned_sample_df, melted_df

# 사용 예시 (주석 처리)
"""
# 함수 사용 예시:
# plot_with_outlier_removal(df0, df1, df2, outlier_method='iqr')
# plot_simple_outlier_removal(df0, df1, df2, outlier_method='zscore')
"""
