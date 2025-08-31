import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, shapiro, levene, bartlett
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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

def check_anova_assumptions(df, columns, alpha=0.05):
    """
    ANOVA 가정들을 검증하는 함수
    
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
    
    # Levene's test (더 강건함)
    groups = list(data_dict.values())
    stat_levene, p_levene = levene(*groups)
    is_homogeneous_levene = p_levene > alpha
    results['homogeneity']['levene'] = {'p_value': p_levene, 'is_homogeneous': is_homogeneous_levene}
    
    print(f"Levene's test: p-value = {p_levene:.4f} {'✅ 등분산' if is_homogeneous_levene else '❌ 등분산 아님'}")
    
    # Bartlett's test (정규분포 가정하에 더 강력함)
    try:
        stat_bartlett, p_bartlett = bartlett(*groups)
        is_homogeneous_bartlett = p_bartlett > alpha
        results['homogeneity']['bartlett'] = {'p_value': p_bartlett, 'is_homogeneous': is_homogeneous_bartlett}
        print(f"Bartlett's test: p-value = {p_bartlett:.4f} {'✅ 등분산' if is_homogeneous_bartlett else '❌ 등분산 아님'}")
    except:
        print("Bartlett's test: 계산 불가")
        results['homogeneity']['bartlett'] = {'p_value': None, 'is_homogeneous': False}
    
    # 3. 이상치 검출
    print("\n=== 이상치 검출 ===")
    for group_name, data in data_dict.items():
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
            print(f"{group_name}: {len(outliers)}개 이상치 발견 - {outliers}")
        else:
            print(f"{group_name}: 이상치 없음")
    
    # 종합 결과
    print(f"\n=== 종합 결과 ===")
    print(f"모든 가정 만족: {'✅ 예' if results['all_assumptions_met'] else '❌ 아니오'}")
    
    return results

def perform_anova(df, columns, alpha=0.05):
    """
    일원배치 ANOVA를 수행하는 함수
    
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
    dict : ANOVA 결과를 담은 딕셔너리
    """
    print("=== 일원배치 ANOVA 분석 ===")
    
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
    
    results = {
        'f_statistic': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'is_significant': is_significant,
        'alpha': alpha
    }
    
    print(f"F-통계량: {f_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"효과크기 (η²): {eta_squared:.4f}")
    print(f"유의성: {'✅ 유의함' if is_significant else '❌ 유의하지 않음'}")
    
    # 효과크기 해석
    if eta_squared < 0.01:
        effect_size_interpretation = "매우 작음"
    elif eta_squared < 0.06:
        effect_size_interpretation = "작음"
    elif eta_squared < 0.14:
        effect_size_interpretation = "중간"
    else:
        effect_size_interpretation = "큼"
    
    print(f"효과크기 해석: {effect_size_interpretation}")
    
    return results

def perform_post_hoc(df, columns, alpha=0.05):
    """
    사후검정 (Tukey's HSD)을 수행하는 함수
    
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
    DataFrame : 사후검정 결과
    """
    print("\n=== 사후검정 (Tukey's HSD) ===")
    
    # DataFrame에서 그룹별 데이터 추출
    data_dict = extract_groups_from_df(df, columns)
    
    if not data_dict:
        print("❌ 분석할 수 있는 유효한 데이터가 없습니다.")
        return None
    
    # 데이터 준비
    all_data = []
    group_labels = []
    
    for group_name, data in data_dict.items():
        all_data.extend(data)
        group_labels.extend([group_name] * len(data))
    
    # Tukey's HSD 수행
    tukey_result = pairwise_tukeyhsd(all_data, group_labels, alpha=alpha)
    
    print(tukey_result)
    
    return tukey_result

def visualize_anova_results(df, columns, anova_results, assumptions_results):
    """
    ANOVA 결과를 시각화하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        분석할 데이터프레임
    columns : list[str]
        분석할 컬럼명들의 리스트
    anova_results : dict
        ANOVA 분석 결과
    assumptions_results : dict
        가정 검증 결과
    """
    # DataFrame에서 그룹별 데이터 추출
    data_dict = extract_groups_from_df(df, columns)
    
    if not data_dict:
        print("❌ 시각화할 수 있는 유효한 데이터가 없습니다.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 박스플롯
    data_for_plot = []
    labels = []
    for group_name, data in data_dict.items():
        data_for_plot.append(data)
        labels.append(group_name)
    
    axes[0, 0].boxplot(data_for_plot, labels=labels)
    axes[0, 0].set_title('Boxplot by Group')
    axes[0, 0].set_ylabel('Values')
    
    # 2. 히스토그램 (정규성 확인)
    for i, (group_name, data) in enumerate(data_dict.items()):
        axes[0, 1].hist(data, alpha=0.7, label=group_name, bins=min(10, len(data)//2))
    axes[0, 1].set_title('Histogram by Group')
    axes[0, 1].set_xlabel('Values')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # 3. Q-Q plot
    for group_name, data in data_dict.items():
        stats.probplot(data, dist="norm", plot=axes[1, 0])
        break  # 첫 번째 그룹만 표시
    axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
    
    # 4. 평균과 신뢰구간
    means = [np.mean(data) for data in data_dict.values()]
    stds = [np.std(data, ddof=1) for data in data_dict.values()]
    group_names = list(data_dict.keys())
    
    axes[1, 1].bar(group_names, means, yerr=stds, capsize=5, alpha=0.7)
    axes[1, 1].set_title('Mean ± Standard Deviation by Group')
    axes[1, 1].set_ylabel('Values')
    
    # ANOVA 결과 텍스트 추가
    if anova_results and anova_results['is_significant']:
        result_text = f"F = {anova_results['f_statistic']:.3f}, p = {anova_results['p_value']:.3f}*"
    elif anova_results:
        result_text = f"F = {anova_results['f_statistic']:.3f}, p = {anova_results['p_value']:.3f}"
    else:
        result_text = "ANOVA 결과 없음"
    
    axes[1, 1].text(0.02, 0.98, result_text, transform=axes[1, 1].transAxes, 
                    verticalalignment='top', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def comprehensive_anova_analysis(df, columns, alpha=0.05):
    """
    종합적인 ANOVA 분석을 수행하는 함수
    
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
    dict : 모든 분석 결과를 담은 딕셔너리
    """
    print("=" * 60)
    print("종합적인 ANOVA 분석 시작")
    print("=" * 60)
    
    # DataFrame에서 그룹별 데이터 추출
    data_dict = extract_groups_from_df(df, columns)
    
    if not data_dict:
        print("❌ 분석할 수 있는 유효한 데이터가 없습니다.")
        return None
    
    # 1. 기술통계
    print("\n📊 기술통계")
    print("-" * 30)
    for group_name, data in data_dict.items():
        print(f"{group_name}:")
        print(f"  n = {len(data)}")
        print(f"  평균 = {np.mean(data):.3f}")
        print(f"  표준편차 = {np.std(data, ddof=1):.3f}")
        print(f"  중앙값 = {np.median(data):.3f}")
        print()
    
    # 2. 가정 검증
    print("\n🔍 ANOVA 가정 검증")
    print("-" * 30)
    assumptions_results = check_anova_assumptions(df, columns, alpha)
    
    # 3. ANOVA 수행
    print("\n📈 ANOVA 분석")
    print("-" * 30)
    anova_results = perform_anova(df, columns, alpha)
    
    # 4. 사후검정 (ANOVA가 유의한 경우)
    post_hoc_results = None
    if anova_results and anova_results['is_significant']:
        post_hoc_results = perform_post_hoc(df, columns, alpha)
    
    # 5. 시각화
    print("\n📊 결과 시각화")
    print("-" * 30)
    visualize_anova_results(df, columns, anova_results, assumptions_results)
    
    # 6. 종합 결과
    print("\n📋 종합 결과 요약")
    print("-" * 30)
    if anova_results:
        print(f"ANOVA 결과: {'유의함' if anova_results['is_significant'] else '유의하지 않음'}")
    else:
        print("ANOVA 결과: 분석 불가")
    
    if assumptions_results:
        print(f"가정 만족: {'예' if assumptions_results['all_assumptions_met'] else '아니오'}")
        
        if not assumptions_results['all_assumptions_met']:
            print("\n⚠️  주의사항:")
            print("- ANOVA 가정이 만족되지 않았습니다.")
            print("- 비모수적 방법(예: Kruskal-Wallis test)을 고려해보세요.")
    else:
        print("가정 검증: 분석 불가")
    
    return {
        'assumptions': assumptions_results,
        'anova': anova_results,
        'post_hoc': post_hoc_results
    }

# 사용 예제
if __name__ == "__main__":
    # 예제 데이터 생성 (DataFrame 형태)
    np.random.seed(42)
    
    # 세 그룹의 데이터 생성
    n_samples = 30
    group1_data = np.random.normal(50, 10, n_samples)  # 평균 50, 표준편차 10, n=30
    group2_data = np.random.normal(55, 10, n_samples)  # 평균 55, 표준편차 10, n=30
    group3_data = np.random.normal(60, 10, n_samples)  # 평균 60, 표준편차 10, n=30
    
    # DataFrame 생성
    sample_df = pd.DataFrame({
        'Group_A': group1_data,
        'Group_B': group2_data,
        'Group_C': group3_data
    })
    
    # 분석할 컬럼명 리스트
    columns_to_analyze = ['Group_A', 'Group_B', 'Group_C']
    
    # 종합 분석 수행
    results = comprehensive_anova_analysis(sample_df, columns_to_analyze, alpha=0.05)