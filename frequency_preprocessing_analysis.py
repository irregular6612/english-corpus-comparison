import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class FrequencyPreprocessingAnalyzer:
    def __init__(self, df, col1, col2):
        """
        빈도수 데이터 전처리 방법별 상관관계 분석 클래스
        
        Parameters:
        df: pandas DataFrame
        col1: 첫 번째 빈도수 컬럼명
        col2: 두 번째 빈도수 컬럼명
        """
        self.df = df.copy()
        self.col1 = col1
        self.col2 = col2
        
        # 결측값 제거
        self.df_clean = self.df[[col1, col2]].dropna()
        
    def apply_preprocessing_methods(self):
        """다양한 전처리 방법 적용"""
        methods = {}
        
        # 1. 원본 데이터
        methods['original'] = {
            'col1': self.df_clean[self.col1],
            'col2': self.df_clean[self.col2]
        }
        
        # 2. 로그 변환 (log1p)
        methods['log_transform'] = {
            'col1': np.log1p(self.df_clean[self.col1]),
            'col2': np.log1p(self.df_clean[self.col2])
        }
        
        # 3. 제곱근 변환
        methods['sqrt_transform'] = {
            'col1': np.sqrt(self.df_clean[self.col1]),
            'col2': np.sqrt(self.df_clean[self.col2])
        }
        
        # 4. 표준화 (StandardScaler)
        scaler_std = StandardScaler()
        std_data = scaler_std.fit_transform(self.df_clean[[self.col1, self.col2]])
        methods['standardization'] = {
            'col1': std_data[:, 0],
            'col2': std_data[:, 1]
        }
        
        # 5. 정규화 (MinMaxScaler)
        scaler_minmax = MinMaxScaler()
        minmax_data = scaler_minmax.fit_transform(self.df_clean[[self.col1, self.col2]])
        methods['normalization'] = {
            'col1': minmax_data[:, 0],
            'col2': minmax_data[:, 1]
        }
        
        # 6. 로버스트 스케일링
        scaler_robust = RobustScaler()
        robust_data = scaler_robust.fit_transform(self.df_clean[[self.col1, self.col2]])
        methods['robust_scaling'] = {
            'col1': robust_data[:, 0],
            'col2': robust_data[:, 1]
        }
        
        # 7. 순위 변환
        methods['rank_transform'] = {
            'col1': self.df_clean[self.col1].rank(),
            'col2': self.df_clean[self.col2].rank()
        }
        
        return methods
    
    def calculate_correlations(self, methods):
        """각 전처리 방법별 상관계수 계산"""
        results = {}
        
        for method_name, data in methods.items():
            col1_data = data['col1']
            col2_data = data['col2']
            
            # 피어슨 상관계수
            pearson_corr, pearson_p = stats.pearsonr(col1_data, col2_data)
            
            # 스피어만 상관계수
            spearman_corr, spearman_p = stats.spearmanr(col1_data, col2_data)
            
            # 켄달 타우
            kendall_corr, kendall_p = stats.kendalltau(col1_data, col2_data)
            
            results[method_name] = {
                'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
                'spearman': {'correlation': spearman_corr, 'p_value': spearman_p},
                'kendall': {'correlation': kendall_corr, 'p_value': kendall_p}
            }
        
        return results
    
    def plot_comparison(self, methods, results):
        """전처리 방법별 비교 시각화"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        method_names = list(methods.keys())
        
        for i, method_name in enumerate(method_names):
            if i >= 8:  # 최대 8개 그래프
                break
                
            data = methods[method_name]
            ax = axes[i]
            
            # 산점도
            ax.scatter(data['col1'], data['col2'], alpha=0.6, s=20)
            
            # 상관계수 표시
            pearson_corr = results[method_name]['pearson']['correlation']
            ax.set_title(f'{method_name}\nPearson r = {pearson_corr:.3f}')
            ax.grid(True, alpha=0.3)
            
            # 축 레이블 간소화
            if i % 4 == 0:  # 왼쪽 열
                ax.set_ylabel('빈도수')
            if i >= 4:  # 아래쪽 행
                ax.set_xlabel('빈도수')
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_comparison(self, results):
        """상관계수 비교 차트"""
        methods = list(results.keys())
        
        # 상관계수 추출
        pearson_corrs = [results[m]['pearson']['correlation'] for m in methods]
        spearman_corrs = [results[m]['spearman']['correlation'] for m in methods]
        kendall_corrs = [results[m]['kendall']['correlation'] for m in methods]
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 상관계수 비교
        x = np.arange(len(methods))
        width = 0.25
        
        ax1.bar(x - width, pearson_corrs, width, label='Pearson', alpha=0.8)
        ax1.bar(x, spearman_corrs, width, label='Spearman', alpha=0.8)
        ax1.bar(x + width, kendall_corrs, width, label='Kendall', alpha=0.8)
        
        ax1.set_xlabel('전처리 방법')
        ax1.set_ylabel('상관계수')
        ax1.set_title('전처리 방법별 상관계수 비교')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # p-value 비교 (피어슨 기준)
        p_values = [results[m]['pearson']['p_value'] for m in methods]
        ax2.bar(methods, p_values, alpha=0.8, color='orange')
        ax2.set_xlabel('전처리 방법')
        ax2.set_ylabel('p-value')
        ax2.set_title('피어슨 상관계수 p-value 비교')
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.axhline(y=0.05, color='red', linestyle='--', label='p=0.05')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_detailed_analysis(self, results):
        """상세 분석 결과 출력"""
        print("=" * 80)
        print("빈도수 데이터 전처리 방법별 상관관계 분석 결과")
        print("=" * 80)
        
        # 기본 통계
        print(f"\n📊 원본 데이터 기본 통계:")
        print(f"데이터 개수: {len(self.df_clean)}")
        print(f"{self.col1} - 평균: {self.df_clean[self.col1].mean():.2f}, 표준편차: {self.df_clean[self.col1].std():.2f}")
        print(f"{self.col2} - 평균: {self.df_clean[self.col2].mean():.2f}, 표준편차: {self.df_clean[self.col2].std():.2f}")
        
        # 각 방법별 결과
        print(f"\n🔍 전처리 방법별 상관계수:")
        print(f"{'방법':<20} {'Pearson':<10} {'Spearman':<10} {'Kendall':<10}")
        print("-" * 60)
        
        for method_name, result in results.items():
            pearson = result['pearson']['correlation']
            spearman = result['spearman']['correlation']
            kendall = result['kendall']['correlation']
            
            print(f"{method_name:<20} {pearson:<10.4f} {spearman:<10.4f} {kendall:<10.4f}")
        
        # 권장사항
        print(f"\n💡 권장사항:")
        print("1. 로그 변환: 빈도수 데이터의 기본적인 전처리 방법")
        print("2. 순위 변환: 이상치에 강하고 비선형 관계도 포착")
        print("3. 로버스트 스케일링: 이상치가 많은 경우 유용")
        print("4. 원본 데이터: 선형 관계가 명확한 경우에만 사용")
        
        print("=" * 80)
    
    def run_full_analysis(self):
        """전체 분석 실행"""
        print("빈도수 데이터 전처리 방법별 상관관계 분석을 시작합니다...")
        
        # 전처리 방법 적용
        methods = self.apply_preprocessing_methods()
        
        # 상관계수 계산
        results = self.calculate_correlations(methods)
        
        # 결과 출력
        self.print_detailed_analysis(results)
        
        # 시각화
        self.plot_comparison(methods, results)
        self.plot_correlation_comparison(results)
        
        return methods, results

# 사용 예시 함수
def analyze_frequency_preprocessing(df, col1, col2):
    """
    빈도수 데이터 전처리 방법별 상관관계 분석
    
    Parameters:
    df: pandas DataFrame
    col1: 첫 번째 빈도수 컬럼명
    col2: 두 번째 빈도수 컬럼명
    """
    analyzer = FrequencyPreprocessingAnalyzer(df, col1, col2)
    return analyzer.run_full_analysis()

# 예시 사용법
if __name__ == "__main__":
    # 예시 데이터 생성 (실제 빈도수 분포와 유사하게)
    np.random.seed(42)
    n_samples = 1000
    
    # 고빈도 단어들 (소수)
    high_freq_words = np.random.poisson(lam=100, size=50)
    
    # 중빈도 단어들
    mid_freq_words = np.random.poisson(lam=20, size=200)
    
    # 저빈도 단어들 (대부분)
    low_freq_words = np.random.poisson(lam=3, size=750)
    
    # 코퍼스1: 빈도수 분포
    freq1 = np.concatenate([high_freq_words, mid_freq_words, low_freq_words])
    
    # 코퍼스2: 코퍼스1과 상관관계가 있지만 스케일이 다른 빈도수
    freq2 = freq1 * 0.6 + np.random.normal(0, 5, len(freq1))
    freq2 = np.maximum(freq2, 0)  # 음수 방지
    
    example_df = pd.DataFrame({
        'corpus1_freq': freq1,
        'corpus2_freq': freq2
    })
    
    print("예시 빈도수 데이터로 전처리 방법별 상관관계 분석 실행:")
    methods, results = analyze_frequency_preprocessing(example_df, 'corpus1_freq', 'corpus2_freq') 