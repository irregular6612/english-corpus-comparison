import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def plot_with_ttest(target_df: pd.DataFrame, HF_df: pd.DataFrame, LF_df: pd.DataFrame):
    """
    코퍼스 간 빈도 비교를 위한 boxplot과 t-test를 함께 수행하는 함수
    
    Parameters:
    -----------
    target_df : pd.DataFrame
        타겟 데이터프레임
    HF_df : pd.DataFrame
        고빈도 데이터프레임
    LF_df : pd.DataFrame
        저빈도 데이터프레임
    value : str
        변환 방법 ('original', 'z-score', 'log', 'zipf_zscore')
    """
    
    target_col = ['RFreq_KF', 'CSAT_RFreq', 'SUBTLRWF', 'RFreq_HAL']
    merged_hf_df = pd.merge(target_df, HF_df, how='inner', on='Word')[['Word'] + target_col]
    merged_hf_df['Freq_type'] = 'HF'
    merged_lf_df = pd.merge(target_df, LF_df, how='inner', on='Word')[['Word'] + target_col]
    merged_lf_df['Freq_type'] = 'LF'
    merged_df = pd.concat([merged_hf_df, merged_lf_df], ignore_index=True)
    
    values = ['original', 'z-score', 'log']
    fig, ax_list = plt.subplots(1, len(values), figsize=(20, 8))

    for i, value in enumerate(values):
        transformed_df = merged_df.copy()
    
        if value == 'original':
            title = 'Original Frequency(per Million Tokens)'
            pass
        elif value == 'z-score':
            title = 'Z-Score'
            transformed_df[target_col] = transformed_df[target_col].apply(lambda x: stats.zscore(x, nan_policy='omit'))
        elif value == 'log':
            title = 'Log10 Frequency(per Million Tokens)'
            transformed_df[target_col] = transformed_df[target_col].apply(lambda x: np.log10(x))
        elif value == 'zipf_zscore':
            title = 'Zipf Frequency(per Billion Tokens)'
            transformed_df[target_col] = transformed_df[target_col].apply(lambda x: np.log10(x))
            transformed_df[target_col] = transformed_df[target_col].apply(lambda x: stats.zscore(x, nan_policy='omit'))
        else:
            raise ValueError(f"Invalid transformation : {value}")
    
        transformed_df = transformed_df.melt(id_vars=['Word', 'Freq_type'], var_name='Corpus', value_name='Freq')
        
        # Boxplot 그리기
        sns.boxplot(transformed_df, x='Freq_type', y='Freq', hue='Corpus', ax=ax_list[i])
        ax_list[i].set_title(title, fontsize=14, fontweight='bold')
        ax_list[i].set_xlabel('')
        ax_list[i].set_ylabel('')
        ax_list[i].legend(title='')
        
        # T-test 수행 및 결과 표시
        add_ttest_annotations(transformed_df, ax_list[i], value)
        
        print(f"\n=== {value.upper()} 변환 결과 ===")
        print(transformed_df.describe())

    plt.tight_layout()
    plt.show()

def add_ttest_annotations(df: pd.DataFrame, ax, value_type: str):
    """
    T-test 결과를 boxplot에 주석으로 추가하는 함수
    
    Parameters:
    -----------
    df : pd.DataFrame
        변환된 데이터프레임
    ax : matplotlib.axes.Axes
        boxplot이 그려진 축
    value_type : str
        변환 타입 (제목용)
    """
    
    # 코퍼스 목록
    corpora = df['Corpus'].unique()
    freq_types = df['Freq_type'].unique()
    
    # ===== 전체 subplot의 데이터 범위 계산 (HF + LF 통합) =====
    # 전체 데이터의 범위를 계산하여 increment 설정
    all_data = df['Freq'].dropna()
    total_data_range = all_data.max() - all_data.min()
    
    # ===== 전체 데이터 범위에 따른 increment 조정 =====
    # increment: 연속된 꺾은 선들 간의 수직 간격 (HF/LF 공통)
    if total_data_range < 1.0:  # 매우 작은 범위 (예: z-score)
        increment = 0.25    # 선 간의 간격 (더 넓게)
    elif total_data_range < 5.0:  # 작은 범위
        increment = 0.5
    elif total_data_range < 10.0:  # 중간 범위
        increment = 0.75
    elif total_data_range < 100:  # 중간 범위
        increment = 8
    elif total_data_range < 1000:  # 중간 범위
        increment = 80
    elif total_data_range < 5000:  # 중간 범위
        increment = 400
    elif total_data_range < 10000:  # 중간 범위
        increment = 300
    else:  # 큰 범위 (예: original frequency)
        increment = 800     # 큰 간격 필요 (값 늘림)
    
    # ===== 전체 유의한 결과들을 수집 =====
    all_significant_results = []
    
    # 각 빈도 타입별로 코퍼스 간 t-test 수행
    for freq_type in freq_types:
        freq_data = df[df['Freq_type'] == freq_type]
        
        # 코퍼스 쌍 조합 생성
        corpus_pairs = list(combinations(corpora, 2))
        
        # 각 쌍에 대해 t-test 수행
        for pair in corpus_pairs:
            corpus1, corpus2 = pair
            
            # 각 코퍼스의 데이터 추출
            data1 = freq_data[freq_data['Corpus'] == corpus1]['Freq'].dropna()
            data2 = freq_data[freq_data['Corpus'] == corpus2]['Freq'].dropna()
            
            # 데이터가 충분한지 확인
            if len(data1) < 2 or len(data2) < 2:
                continue
            
            # T-test 수행
            try:
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                
                # 유의성 표시 결정
                significance = get_significance_symbol(p_value)
                
                if significance:  # 유의한 경우에만 저장
                    # ===== 꺾은 선의 X축 위치 계산 =====
                    # freq_type: 'HF' 또는 'LF' 그룹의 인덱스 (0 또는 1)
                    x_pos = freq_types.tolist().index(freq_type)
                    
                    # corpus_positions: 각 코퍼스의 상대적 위치 (0, 1, 2, 3)
                    # boxplot에서 코퍼스들이 가로로 나열된 순서
                    corpus_positions = {corpus: idx for idx, corpus in enumerate(corpora)}
                    y1_pos = corpus_positions[corpus1]  # 첫 번째 코퍼스의 위치
                    y2_pos = corpus_positions[corpus2]  # 두 번째 코퍼스의 위치
                    
                    # x_start, x_end: 꺾은 선의 시작점과 끝점 X좌표
                    # x_pos - 0.3: HF/LF 그룹의 중심에서 왼쪽으로 0.3 이동
                    # y1_pos * 0.2: 각 코퍼스 간의 간격 (0.2 단위로 배치)
                    x_start = x_pos - 0.3 + (y1_pos * 0.2)  # 첫 번째 코퍼스 위의 X좌표
                    x_end = x_pos - 0.3 + (y2_pos * 0.2)    # 두 번째 코퍼스 위의 X좌표
                    
                    # ===== 꺾은 선의 Y축 위치 계산 =====
                    # y_data1, y_data2: 각 코퍼스의 빈도 데이터
                    y_data1 = freq_data[freq_data['Corpus'] == corpus1]['Freq']
                    y_data2 = freq_data[freq_data['Corpus'] == corpus2]['Freq']
                    
                    # y_max1, y_max2: 각 boxplot의 상단 경계 (outlier 제외한 최대값)
                    # Q3 + 1.5*IQR 공식으로 계산 (boxplot의 whisker 끝점)
                    y_max1 = y_data1.quantile(0.75) + 1.5 * (y_data1.quantile(0.75) - y_data1.quantile(0.25))
                    y_max2 = y_data2.quantile(0.75) + 1.5 * (y_data2.quantile(0.75) - y_data2.quantile(0.25))
                    
                    # base_y: 꺾은 선의 기본 Y좌표 (boxplot 위의 여유 공간)
                    # max(y_max1, y_max2): 두 boxplot 중 더 높은 것 기준
                    # (y_max1 + y_max2) * 0.05: 추가 여백 (데이터 범위의 5%로 줄임)
                    base_y = max(y_max1, y_max2) + (y_max1 + y_max2) * 0.05
                    
                    all_significant_results.append({
                        'corpus1': corpus1,
                        'corpus2': corpus2,
                        'x_start': x_start,
                        'x_end': x_end,
                        'base_y': base_y,
                        'p_value': p_value,
                        'significance': significance,
                        'freq_type': freq_type  # HF/LF 구분을 위해 추가
                    })
                    
            except Exception as e:
                print(f"T-test 오류 ({corpus1} vs {corpus2}): {e}")
    
    # ===== HF와 LF를 구분하여 각각 처리 =====
    if all_significant_results:
        # HF와 LF 결과를 그룹별로 분리
        hf_results = [r for r in all_significant_results if r['freq_type'] == 'HF']
        lf_results = [r for r in all_significant_results if r['freq_type'] == 'LF']
        
        # ===== 전체 유의한 결과 개수에 따른 추가 조정 =====
        total_num_results = len(all_significant_results)
        if total_num_results > 3:
            increment *= 1.5  # 결과가 많으면 간격을 50% 늘림 (더 넓게)
        elif total_num_results > 5:
            increment *= 2.0  # 결과가 5개 초과면 간격을 100% 늘림 (더 넓게)
        
        # ===== HF 그룹 내에서 꺾은 선 그리기 =====
        if hf_results:
            # HF 결과를 p값 순으로 정렬
            hf_results.sort(key=lambda x: x['p_value'])
            
            # ===== HF 그룹의 데이터 범위에 따른 base_offset 계산 =====
            hf_data = df[df['Freq_type'] == 'HF']['Freq'].dropna()
            hf_data_range = hf_data.max() - hf_data.min()
            
            # HF 그룹의 데이터 범위에 따라 base_offset 설정
            if hf_data_range < 1.0:
                hf_base_offset = 0.02
            elif hf_data_range < 5.0:
                hf_base_offset = 0.02
            elif hf_data_range < 10.0:
                hf_base_offset = 0.02
            elif hf_data_range < 100:
                hf_base_offset = 1
            elif hf_data_range < 1000:
                hf_base_offset = 10
            elif hf_data_range < 5000:
                hf_base_offset = 50
            elif hf_data_range < 10000:
                hf_base_offset = 10
            else:
                hf_base_offset = 100
            
            for i, result in enumerate(hf_results):
                # HF 그룹 내에서의 높이 오프셋 계산
                height_offset = hf_base_offset + (i * increment)
                y_line = result['base_y'] + height_offset
                
                # ===== 꺾은 선의 수직 부분 길이 조정 =====
                if value_type == 'original':
                    vertical_length = 150
                else:
                    vertical_length = 0.08
                
                # ===== 꺾은 선 그리기 =====
                ax.plot([result['x_start'], result['x_start'], result['x_end'], result['x_end']], 
                       [y_line, y_line + vertical_length, y_line + vertical_length, y_line], 
                       'k-', linewidth=1, alpha=0.7)
                
                # ===== 유의성 기호와 p값 표시 =====
                if result['p_value'] < 0.001:
                    p_display = '(p<0.001)'
                else:
                    p_display = f'(p={result["p_value"]:.3f})'
                
                significance_text = f"{result['significance']}{p_display}"
                ax.text((result['x_start'] + result['x_end']) / 2, y_line + vertical_length + 0.04, significance_text, 
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # ===== LF 그룹 내에서 꺾은 선 그리기 =====
        if lf_results:
            # LF 결과를 p값 순으로 정렬
            lf_results.sort(key=lambda x: x['p_value'])
            
            # ===== LF 그룹의 데이터 범위에 따른 base_offset 계산 =====
            lf_data = df[df['Freq_type'] == 'LF']['Freq'].dropna()
            lf_data_range = lf_data.max() - lf_data.min()
            
            # LF 그룹의 데이터 범위에 따라 base_offset 설정
            if lf_data_range < 1.0:
                lf_base_offset = 0.02
            elif lf_data_range < 5.0:
                lf_base_offset = 0.02
            elif lf_data_range < 10.0:
                lf_base_offset = 0.02
            elif lf_data_range < 100:
                lf_base_offset = 1
            elif lf_data_range < 1000:
                lf_base_offset = 10
            elif lf_data_range < 5000:
                lf_base_offset = 50
            elif lf_data_range < 10000:
                lf_base_offset = 10
            else:
                lf_base_offset = 100
            
            for i, result in enumerate(lf_results):
                # LF 그룹 내에서의 높이 오프셋 계산
                height_offset = lf_base_offset + (i * increment)
                y_line = result['base_y'] + height_offset
                
                # ===== 꺾은 선의 수직 부분 길이 조정 =====
                if value_type == 'original':
                    vertical_length = 150
                else:
                    vertical_length = 0.08
                
                # ===== 꺾은 선 그리기 =====
                ax.plot([result['x_start'], result['x_start'], result['x_end'], result['x_end']], 
                       [y_line, y_line + vertical_length, y_line + vertical_length, y_line], 
                       'k-', linewidth=1, alpha=0.7)
                
                # ===== 유의성 기호와 p값 표시 =====
                if result['p_value'] < 0.001:
                    p_display = '(p<0.001)'
                else:
                    p_display = f'(p={result["p_value"]:.3f})'
                
                significance_text = f"{result['significance']}{p_display}"
                ax.text((result['x_start'] + result['x_end']) / 2, y_line + vertical_length + 0.04, significance_text, 
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
            
             

def get_significance_symbol(p_value: float) -> str:
    """
    p값을 유의성 기호로 변환하는 함수
    
    Parameters:
    -----------
    p_value : float
        p값
    
    Returns:
    --------
    str
        유의성 기호 (* p<0.05, ** p<0.01, *** p<0.001)
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

def print_ttest_summary(target_df: pd.DataFrame, HF_df: pd.DataFrame, LF_df: pd.DataFrame, value='z-score'):
    """
    T-test 결과를 요약하여 출력하는 함수
    
    Parameters:
    -----------
    target_df : pd.DataFrame
        타겟 데이터프레임
    HF_df : pd.DataFrame
        고빈도 데이터프레임
    LF_df : pd.DataFrame
        저빈도 데이터프레임
    value : str
        변환 방법
    """
    
    target_col = ['RFreq_KF', 'CSAT_RFreq', 'SUBTLRWF', 'RFreq_HAL']
    merged_hf_df = pd.merge(target_df, HF_df, how='inner', on='Word')[['Word'] + target_col]
    merged_hf_df['Freq_type'] = 'HF'
    merged_lf_df = pd.merge(target_df, LF_df, how='inner', on='Word')[['Word'] + target_col]
    merged_lf_df['Freq_type'] = 'LF'
    merged_df = pd.concat([merged_hf_df, merged_lf_df], ignore_index=True)
    
    # 데이터 변환
    transformed_df = merged_df.copy()
    
    if value == 'z-score':
        transformed_df[target_col] = transformed_df[target_col].apply(lambda x: stats.zscore(x, nan_policy='omit'))
    elif value == 'log':
        transformed_df[target_col] = transformed_df[target_col].apply(lambda x: np.log10(x))
    elif value == 'zipf_zscore':
        transformed_df[target_col] = transformed_df[target_col].apply(lambda x: np.log10(x))
        transformed_df[target_col] = transformed_df[target_col].apply(lambda x: stats.zscore(x, nan_policy='omit'))
    
    corpora = target_col
    freq_types = ['HF', 'LF']
    
    print(f"\n=== {value.upper()} 변환 T-Test 결과 요약 ===")
    
    for freq_type in freq_types:
        print(f"\n[{freq_type} 그룹]")
        freq_data = transformed_df[transformed_df['Freq_type'] == freq_type]
        
        corpus_pairs = list(combinations(corpora, 2))
        
        for pair in corpus_pairs:
            corpus1, corpus2 = pair
            
            data1 = freq_data[corpus1].dropna()
            data2 = freq_data[corpus2].dropna()
            
            if len(data1) < 2 or len(data2) < 2:
                continue
            
            try:
                t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
                significance = get_significance_symbol(p_value)
                
                print(f"  {corpus1} vs {corpus2}:")
                print(f"    t-statistic: {t_stat:.4f}")
                print(f"    p-value: {p_value:.4f} {significance}")
                print(f"    mean1: {data1.mean():.4f}, mean2: {data2.mean():.4f}")
                print()
                
            except Exception as e:
                print(f"  {corpus1} vs {corpus2}: 오류 - {e}")
