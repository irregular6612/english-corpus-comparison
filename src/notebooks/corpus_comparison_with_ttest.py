import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, shapiro, levene, bartlett
from scipy.stats import t as student_t
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def plot_with_anova(target_df: pd.DataFrame, HF_df: pd.DataFrame, LF_df: pd.DataFrame):
    """
    코퍼스 간 빈도 비교를 위한 boxplot과 ANOVA를 함께 수행하는 함수
    
    Parameters:
    -----------
    target_df : pd.DataFrame
        타겟 데이터프레임
    HF_df : pd.DataFrame
        고빈도 데이터프레임
    LF_df : pd.DataFrame
        저빈도 데이터프레임
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
        
        # 그룹 평균과 95% 신뢰구간 오버레이
        add_group_mean_ci_overlays(transformed_df, ax_list[i])
        
        # ANOVA 수행 및 결과 표시
        add_anova_annotations(transformed_df, ax_list[i], value)
        
        print(f"\n=== {value.upper()} 변환 결과 ===")
        print(transformed_df.describe())

    plt.tight_layout()
    plt.show()

def extract_groups_from_df(df, columns):
    """
    DataFrame에서 지정된 컬럼들을 그룹별 데이터로 추출하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        분석할 데이터프레임
    columns : list[str]
        분석할 컬럼명들의 리스트
    
    Returns:
    --------
    dict : 그룹별 데이터를 담은 딕셔너리
    """
    data_dict = {}
    
    for column in columns:
        if column not in df.columns:
            print(f"⚠️  경고: 컬럼 '{column}'이 데이터프레임에 존재하지 않습니다.")
            continue
        
        # NaN 값 제거
        clean_data = df[column].dropna().tolist()
        if len(clean_data) == 0:
            print(f"⚠️  경고: 컬럼 '{column}'에 유효한 데이터가 없습니다.")
            continue
            
        data_dict[column] = clean_data
    
    return data_dict

def perform_anova_and_tukey(df, columns, alpha=0.05):
    """
    일원배치 ANOVA와 Tukey's HSD를 수행하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        분석할 데이터프레임
    columns : list[str]
        분석할 컬럼명들의 리스트
    alpha : float
        유의수준 (기본값: 0.05)
    
    Returns:
    --------
    dict : ANOVA와 Tukey 결과를 담은 딕셔너리
    """
    # DataFrame에서 그룹별 데이터 추출
    data_dict = extract_groups_from_df(df, columns)
    
    if not data_dict:
        print("❌ 분석할 수 있는 유효한 데이터가 없습니다.")
        return None
    
    # 데이터 준비
    groups = list(data_dict.values())
    group_names = list(data_dict.keys())
    
    # ANOVA 수행
    f_stat, p_value = f_oneway(*groups)
    
    # 효과크기 계산 (eta-squared)
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    # SS_between (그룹간 제곱합)
    ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
    
    # SS_total (전체 제곱합)
    ss_total = sum((x - grand_mean)**2 for x in all_data)
    
    # eta-squared
    eta_squared = ss_between / ss_total
    
    # 결과 해석
    is_significant = p_value < alpha
    
    anova_results = {
        'f_statistic': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'is_significant': is_significant,
        'alpha': alpha
    }
    
    # Tukey's HSD 수행 (ANOVA가 유의한 경우)
    tukey_results = None
    if is_significant:
        # 데이터 준비
        all_data = []
        group_labels = []
        
        for group_name, data in data_dict.items():
            all_data.extend(data)
            group_labels.extend([group_name] * len(data))
        
        # Tukey's HSD 수행
        tukey_results = pairwise_tukeyhsd(all_data, group_labels, alpha=alpha)
    
    return {
        'anova': anova_results,
        'tukey': tukey_results
    }

def add_anova_annotations(df: pd.DataFrame, ax, value_type: str):
    """
    ANOVA와 Tukey 결과를 boxplot에 주석으로 추가하는 함수
    
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
    all_data = df['Freq'].dropna()
    total_data_range = all_data.max() - all_data.min()
    
    # ===== 전체 데이터 범위에 따른 increment 조정 (전역 기본값) =====
    if total_data_range < 1.0:
        increment = 0.25
    elif total_data_range < 5.0:
        increment = 0.5
    elif total_data_range < 10.0:
        increment = 0.75
    elif total_data_range < 100:
        increment = 8
    elif total_data_range < 1000:
        increment = 80
    elif total_data_range < 5000:
        increment = 400
    elif total_data_range < 10000:
        increment = 300
    else:
        increment = 800
    
    # ===== 각 빈도 타입별로 ANOVA 및 Tukey 수행 =====
    all_significant_results = []
    stat_pairs = []
    stat_pvalues = []
    
    for freq_type in freq_types:
        freq_data = df[df['Freq_type'] == freq_type]
        
        # 각 코퍼스의 데이터를 컬럼으로 변환
        corpus_data = {}
        for corpus in corpora:
            corpus_data[corpus] = freq_data[freq_data['Corpus'] == corpus]['Freq'].dropna().tolist()
        
        # DataFrame으로 변환 (길이 패딩)
        corpus_series = {k: pd.Series(v) for k, v in corpus_data.items()}
        corpus_df = pd.DataFrame(corpus_series)
        
        # ANOVA 및 Tukey 수행
        results = perform_anova_and_tukey(corpus_df, list(corpus_data.keys()))
        
        if results and results['anova']['is_significant'] and results['tukey'] is not None:
            # Tukey 결과 테이블 파싱
            tukey_result = results['tukey']
            tukey_df = pd.DataFrame(tukey_result._results_table.data[1:], 
                                   columns=tukey_result._results_table.data[0])
            significant_pairs = tukey_df[tukey_df['reject'] == True]
            
            for idx, row in significant_pairs.iterrows():
                corpus1 = row['group1']
                corpus2 = row['group2']
                p_value = float(row['p-adj'])
                # Tukey CI (mean diff의 CI)
                lower_ci = float(row['lower'])
                upper_ci = float(row['upper'])
                
                # 유의성 기호 결정
                significance = get_significance_symbol(p_value)
                
                if significance:  # 유의한 경우에만 저장
                    # ===== 꺾은 선의 X축 위치 계산 =====
                    x_pos = freq_types.tolist().index(freq_type)
                    corpus_positions = {corpus: idx for idx, corpus in enumerate(corpora)}
                    y1_pos = corpus_positions[corpus1]
                    y2_pos = corpus_positions[corpus2]
                    x_start = x_pos - 0.3 + (y1_pos * 0.2)
                    x_end = x_pos - 0.3 + (y2_pos * 0.2)
                    
                    # ===== 꺾은 선의 Y축 위치 계산 (box 상단: Q3 기준) =====
                    y_data1 = freq_data[freq_data['Corpus'] == corpus1]['Freq']
                    y_data2 = freq_data[freq_data['Corpus'] == corpus2]['Freq']
                    y_q3_1 = y_data1.quantile(0.75)
                    y_q3_2 = y_data2.quantile(0.75)
                    base_y = max(y_q3_1, y_q3_2)
                    
                    all_significant_results.append({
                        'corpus1': corpus1,
                        'corpus2': corpus2,
                        'x_start': x_start,
                        'x_end': x_end,
                        'base_y': base_y,
                        'p_value': p_value,
                        'ci_lower': lower_ci,
                        'ci_upper': upper_ci,
                        'significance': significance,
                        'freq_type': freq_type
                    })
                    
                    # statannotations용 페어 수집
                    stat_pairs.append(((freq_type, corpus1), (freq_type, corpus2)))
                    stat_pvalues.append(p_value)
    
    # ===== statannotations 사용 가능하면 우선 사용 (없으면 수동으로 진행) =====
    try:
        if stat_pairs:
            from statannotations.Annotator import Annotator
            annot = Annotator(ax, stat_pairs, data=df, x='Freq_type', y='Freq', hue='Corpus')
            annot.configure(test=None, text_format='star', pvalues=stat_pvalues, show_test_name=False)
            annot.annotate()
            return
    except Exception:
        # 라이브러리 미설치/버전 이슈 시 수동으로 진행
        pass
    
    # ===== HF와 LF를 구분하여 각각 수동으로 꺾은 선 그리기 =====
    if all_significant_results:
        # 그룹 분리
        hf_results = [r for r in all_significant_results if r['freq_type'] == 'HF']
        lf_results = [r for r in all_significant_results if r['freq_type'] == 'LF']
        
        # 레벨 배치 함수 (x-구간 겹침 최소화)
        def assign_bracket_levels(results_list, tol=0.02):
            # 결과를 좌측 x, 우측 x, 폭 기준으로 정렬(왼->오, 짧은폭 우선)
            indexed = list(enumerate(results_list))
            indexed.sort(key=lambda it: (min(it[1]['x_start'], it[1]['x_end']), abs(it[1]['x_end'] - it[1]['x_start'])))
            levels = [None] * len(results_list)
            level_spans = []  # 각 레벨별 [(start, end)]
            for idx, res in indexed:
                a = min(res['x_start'], res['x_end'])
                b = max(res['x_start'], res['x_end'])
                placed = False
                for level, spans in enumerate(level_spans):
                    # 기존 스팬과 겹치지 않으면 같은 레벨에 배치
                    if all(b < s[0] - tol or a > s[1] + tol for s in spans):
                        spans.append((a, b))
                        levels[idx] = level
                        placed = True
                        break
                if not placed:
                    level_spans.append([(a, b)])
                    levels[idx] = len(level_spans) - 1
            return levels
        
        # ===== HF 그룹 =====
        if hf_results:
            hf_results.sort(key=lambda x: x['p_value'])
            # 현재 서브플롯 y축 범위 기반 퍼센트 계산
            y_min, y_max = ax.get_ylim()
            axis_range = max(y_max - y_min, 1e-12)
            base_margin_hf = 0.02 * axis_range   # 2%
            increment_hf = 0.06 * axis_range     # 6%
            vertical_len_hf = 0.02 * axis_range  # 2%
            text_offset_hf = 0.015 * axis_range  # 1.5%
            if value_type != 'original':
                vertical_len_hf = max(vertical_len_hf, 0.06)
                text_offset_hf = max(text_offset_hf, 0.03)
            increment_hf = max(increment_hf, vertical_len_hf * 2.0 + text_offset_hf * 1.5)
            
            # 레벨 할당으로 겹침 방지
            hf_levels = assign_bracket_levels(hf_results, tol=0.02)
            for i, result in enumerate(hf_results):
                level = hf_levels[i]
                y_line = result['base_y'] + base_margin_hf + (level * increment_hf)
                ax.plot([result['x_start'], result['x_start'], result['x_end'], result['x_end']], 
                       [y_line, y_line + vertical_len_hf, y_line + vertical_len_hf, y_line], 
                       'k-', linewidth=1, alpha=0.8)
                p_display = '(p<0.001)' if result['p_value'] < 0.001 else f"(p={result['p_value']:.3f})"
                significance_text = f"{result['significance']}{p_display}"
                ax.text((result['x_start'] + result['x_end']) / 2, y_line + vertical_len_hf + text_offset_hf, significance_text, 
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # ===== LF 그룹 =====
        if lf_results:
            lf_results.sort(key=lambda x: x['p_value'])
            # 현재 서브플롯 y축 범위 기반 퍼센트 계산
            y_min, y_max = ax.get_ylim()
            axis_range = max(y_max - y_min, 1e-12)
            base_margin_lf = 0.02 * axis_range   # 2%
            increment_lf = 0.06 * axis_range     # 6%
            vertical_len_lf = 0.02 * axis_range  # 2%
            text_offset_lf = 0.015 * axis_range  # 1.5%
            if value_type != 'original':
                vertical_len_lf = max(vertical_len_lf, 0.06)
                text_offset_lf = max(text_offset_lf, 0.03)
            increment_lf = max(increment_lf, vertical_len_lf * 2.0 + text_offset_lf * 1.5)
            
            lf_levels = assign_bracket_levels(lf_results, tol=0.02)
            for i, result in enumerate(lf_results):
                level = lf_levels[i]
                y_line = result['base_y'] + base_margin_lf + (level * increment_lf)
                ax.plot([result['x_start'], result['x_start'], result['x_end'], result['x_end']], 
                       [y_line, y_line + vertical_len_lf, y_line + vertical_len_lf, y_line], 
                       'k-', linewidth=1, alpha=0.8)
                p_display = '(p<0.001)' if result['p_value'] < 0.001 else f"(p={result['p_value']:.3f})"
                significance_text = f"{result['significance']}{p_display}"
                ax.text((result['x_start'] + result['x_end']) / 2, y_line + vertical_len_lf + text_offset_lf, significance_text, 
                       ha='center', va='bottom', fontsize=12, fontweight='bold')

def _mean_ci(series: pd.Series, alpha: float = 0.05):
    values = pd.to_numeric(series.dropna(), errors='coerce')
    values = values.dropna().values
    n = len(values)
    if n < 2:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(n))
    tcrit = float(student_t.ppf(1 - alpha / 2, df=n - 1))
    half = tcrit * sem
    return (mean, mean - half, mean + half)

def add_group_mean_ci_overlays(df: pd.DataFrame, ax):
    """
    박스플롯 위에 (Freq_type, Corpus) 그룹별 평균과 95% CI를 errorbar로 오버레이
    """
    corpora = df['Corpus'].unique()
    freq_types = df['Freq_type'].unique()
    for freq_type in freq_types:
        sub = df[df['Freq_type'] == freq_type]
        x_pos = freq_types.tolist().index(freq_type)
        for idx, corpus in enumerate(corpora):
            vals = sub[sub['Corpus'] == corpus]['Freq']
            mean, lower, upper = _mean_ci(vals, alpha=0.05)
            if np.isnan(mean):
                continue
            x = x_pos - 0.3 + (idx * 0.2)
            yerr_lower = mean - lower
            yerr_upper = upper - mean
            ax.errorbar(x, mean, yerr=[[yerr_lower], [yerr_upper]], fmt='o', color='black',
                        capsize=4, markersize=4, linewidth=1, zorder=5)

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

def print_anova_summary(target_df: pd.DataFrame, HF_df: pd.DataFrame, LF_df: pd.DataFrame, value='z-score'):
    """
    ANOVA 결과를 요약하여 출력하는 함수
    
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
    
    print(f"\n=== {value.upper()} 변환 ANOVA 결과 요약 ===")
    
    for freq_type in freq_types:
        print(f"\n[{freq_type} 그룹]")
        freq_data = transformed_df[transformed_df['Freq_type'] == freq_type]
        
        # 각 코퍼스의 데이터를 컬럼으로 변환
        corpus_data = {}
        for corpus in corpora:
            if corpus in freq_data.columns:
                corpus_data[corpus] = freq_data[corpus].dropna().tolist()
            else:
                corpus_data[corpus] = []
        
        # 리스트 길이 불일치 방지: 각 리스트를 Series로 변환하여 NaN 패딩 후 DataFrame 생성
        corpus_series = {k: pd.Series(v) for k, v in corpus_data.items()}
        corpus_df = pd.DataFrame(corpus_series)
        
        # 가정 검증 출력
        print("  🔍 ANOVA 가정 검증")
        assumptions = check_anova_assumptions(corpus_df, list(corpus_data.keys()), alpha=0.05)
        
        # ANOVA 및 Tukey 수행
        results = perform_anova_and_tukey(corpus_df, list(corpus_data.keys()))
        
        if results:
            anova_result = results['anova']
            print(f"  ANOVA 결과:")
            print(f"    F-statistic: {anova_result['f_statistic']:.4f}")
            print(f"    p-value: {anova_result['p_value']:.4f}")
            print(f"    eta-squared: {anova_result['eta_squared']:.4f}")
            print(f"    유의성: {'유의함' if anova_result['is_significant'] else '유의하지 않음'}")
            
            if anova_result['is_significant'] and results['tukey'] is not None:
                print(f"  Tukey's HSD 결과:")
                tukey_result = results['tukey']
                
                # Tukey 결과를 DataFrame으로 변환하여 처리
                tukey_df = pd.DataFrame(tukey_result._results_table.data[1:], 
                                       columns=tukey_result._results_table.data[0])
                
                # Tukey 결과를 보기 좋게 출력
                print("    그룹 쌍별 비교:")
                for idx, row in tukey_df.iterrows():
                    group1 = row['group1']
                    group2 = row['group2']
                    p_value = float(row['p-adj'])
                    reject = row['reject']
                    meandiff = float(row['meandiff'])
                    lower = float(row['lower'])
                    upper = float(row['upper'])
                    significance = get_significance_symbol(p_value)
                    
                    print(f"      {group1} vs {group2}: diff = {meandiff:.3f}, CI[{lower:.3f},{upper:.3f}], p = {p_value:.4f} {significance} {'(유의함)' if reject else '(유의하지 않음)'}")
        else:
            print("  ANOVA 분석 불가")
        
        print()

def check_anova_assumptions(df, columns, alpha=0.05):
    """
    ANOVA 가정들을 검증하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        분석할 데이터프레임 (각 컬럼이 그룹)
    columns : list[str]
        분석할 컬럼명들의 리스트
    alpha : float
        유의수준 (기본값: 0.05)
    
    Returns:
    --------
    dict : 검증 결과를 담은 딕셔너리
    """
    # DataFrame에서 그룹별 데이터 추출
    data_dict = extract_groups_from_df(df, columns)
    
    if not data_dict:
        print("❌ 분석할 수 있는 유효한 데이터가 없습니다.")
        return None
    
    results = {
        'normality': {},
        'homogeneity': {},
        'outliers': {},
        'all_assumptions_met': True
    }
    
    # 1. 정규성 검정 (Shapiro-Wilk test)
    print("=== 정규성 검정 (Shapiro-Wilk test) ===")
    for group_name, data in data_dict.items():
        if len(data) < 3:
            print(f"⚠️  {group_name}: 샘플 수가 너무 적습니다 (n={len(data)})")
            results['normality'][group_name] = {'p_value': None, 'is_normal': False}
            results['all_assumptions_met'] = False
            continue
            
        stat, p_value = shapiro(data)
        is_normal = p_value > alpha
        results['normality'][group_name] = {'p_value': p_value, 'is_normal': is_normal}
        status = "✅ 정규분포" if is_normal else "❌ 정규분포 아님"
        print(f"{group_name}: p-value = {p_value:.4f} {status}")
        if not is_normal:
            results['all_assumptions_met'] = False
    
    # 2. 등분산 검정
    print("\n=== 등분산 검정 ===")
    groups = list(data_dict.values())
    try:
        stat_levene, p_levene = levene(*groups)
        is_homogeneous_levene = p_levene > alpha
        results['homogeneity']['levene'] = {'p_value': p_levene, 'is_homogeneous': is_homogeneous_levene}
        print(f"Levene's test: p-value = {p_levene:.4f} {'✅ 등분산' if is_homogeneous_levene else '❌ 등분산 아님'}")
    except Exception:
        print("Levene's test: 계산 불가")
        results['homogeneity']['levene'] = {'p_value': None, 'is_homogeneous': False}
        results['all_assumptions_met'] = False
    
    try:
        stat_bartlett, p_bartlett = bartlett(*groups)
        is_homogeneous_bartlett = p_bartlett > alpha
        results['homogeneity']['bartlett'] = {'p_value': p_bartlett, 'is_homogeneous': is_homogeneous_bartlett}
        print(f"Bartlett's test: p-value = {p_bartlett:.4f} {'✅ 등분산' if is_homogeneous_bartlett else '❌ 등분산 아님'}")
    except Exception:
        print("Bartlett's test: 계산 불가")
        results['homogeneity']['bartlett'] = {'p_value': None, 'is_homogeneous': False}
        results['all_assumptions_met'] = False
    
    # 3. 이상치 검출 (IQR 기준)
    print("\n=== 이상치 검출 ===")
    for group_name, data in data_dict.items():
        if len(data) == 0:
            results['outliers'][group_name] = {
                'outliers': [], 'outlier_indices': [], 'outlier_count': 0
            }
            print(f"{group_name}: 데이터 없음")
            continue
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        outlier_indices = [i for i, x in enumerate(data) if x < lower_bound or x > upper_bound]
        results['outliers'][group_name] = {
            'outliers': outliers,
            'outlier_indices': outlier_indices,
            'outlier_count': len(outliers)
        }
        if outliers:
            print(f"{group_name}: {len(outliers)}개 이상치 발견")
        else:
            print(f"{group_name}: 이상치 없음")
    
    print(f"\n=== 종합 결과 ===")
    print(f"모든 가정 만족: {'✅ 예' if results['all_assumptions_met'] else '❌ 아니오'}")
    return results
