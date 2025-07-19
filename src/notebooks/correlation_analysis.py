import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS용
plt.rcParams['axes.unicode_minus'] = False

class FrequencyCorrelationAnalyzer:
    def __init__(self, df, col1, col2):
        """
        단어 빈도수 컬럼 간 상관관계 분석 클래스
        
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
        
    def calculate_correlation(self):
        """상관계수 계산"""
        # 피어슨 상관계수 (선형 관계)
        pearson_corr, pearson_p = stats.pearsonr(self.df_clean[self.col1], self.df_clean[self.col2])
        
        # 스피어만 상관계수 (순위 관계)
        spearman_corr, spearman_p = stats.spearmanr(self.df_clean[self.col1], self.df_clean[self.col2])
        
        # 켄달 타우 (순위 관계)
        kendall_corr, kendall_p = stats.kendalltau(self.df_clean[self.col1], self.df_clean[self.col2])
        
        return {
            'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
            'spearman': {'correlation': spearman_corr, 'p_value': spearman_p},
            'kendall': {'correlation': kendall_corr, 'p_value': kendall_p}
        }
    
    def plot_scatter(self, figsize=(10, 8)):
        """산점도 그리기"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 원본 데이터 산점도
        ax1.scatter(self.df_clean[self.col1], self.df_clean[self.col2], alpha=0.6)
        ax1.set_xlabel(self.col1)
        ax1.set_ylabel(self.col2)
        ax1.set_title(f'{self.col1} vs {self.col2} (원본 데이터)')
        ax1.grid(True, alpha=0.3)
        
        # 로그 변환된 데이터 산점도 (빈도수 데이터에 유용)
        log_col1 = np.log1p(self.df_clean[self.col1])  # log(1+x) 변환
        log_col2 = np.log1p(self.df_clean[self.col2])
        
        ax2.scatter(log_col1, log_col2, alpha=0.6, color='orange')
        ax2.set_xlabel(f'log(1 + {self.col1})')
        ax2.set_ylabel(f'log(1 + {self.col2})')
        ax2.set_title(f'{self.col1} vs {self.col2} (로그 변환)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_heatmap(self, figsize=(8, 6)):
        """상관계수 히트맵"""
        corr_matrix = self.df_clean.corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f')
        plt.title('상관계수 히트맵')
        plt.show()
    
    def plot_distribution(self, figsize=(12, 5)):
        """분포 시각화"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # 첫 번째 컬럼 분포
        ax1.hist(self.df_clean[self.col1], bins=30, alpha=0.7, color='skyblue')
        ax1.set_xlabel(self.col1)
        ax1.set_ylabel('빈도')
        ax1.set_title(f'{self.col1} 분포')
        ax1.grid(True, alpha=0.3)
        
        # 두 번째 컬럼 분포
        ax2.hist(self.df_clean[self.col2], bins=30, alpha=0.7, color='lightcoral')
        ax2.set_xlabel(self.col2)
        ax2.set_ylabel('빈도')
        ax2.set_title(f'{self.col2} 분포')
        ax2.grid(True, alpha=0.3)
        
        # 박스플롯
        ax3.boxplot([self.df_clean[self.col1], self.df_clean[self.col2]], 
                   labels=[self.col1, self.col2])
        ax3.set_ylabel('빈도수')
        ax3.set_title('박스플롯 비교')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def print_summary(self):
        """분석 결과 요약 출력"""
        print("=" * 60)
        print("단어 빈도수 상관관계 분석 결과")
        print("=" * 60)
        
        # 기본 통계
        print(f"\n📊 기본 통계:")
        print(f"데이터 개수: {len(self.df_clean)}")
        print(f"{self.col1} - 평균: {self.df_clean[self.col1].mean():.2f}, 표준편차: {self.df_clean[self.col1].std():.2f}")
        print(f"{self.col2} - 평균: {self.df_clean[self.col2].mean():.2f}, 표준편차: {self.df_clean[self.col2].std():.2f}")
        
        # 상관계수
        corr_results = self.calculate_correlation()
        
        print(f"\n🔗 상관계수:")
        print(f"피어슨 상관계수: {corr_results['pearson']['correlation']:.4f} (p-value: {corr_results['pearson']['p_value']:.4f})")
        print(f"스피어만 상관계수: {corr_results['spearman']['correlation']:.4f} (p-value: {corr_results['spearman']['p_value']:.4f})")
        print(f"켄달 타우: {corr_results['kendall']['correlation']:.4f} (p-value: {corr_results['kendall']['p_value']:.4f})")
        
        # 해석
        print(f"\n📝 해석:")
        pearson_corr = corr_results['pearson']['correlation']
        pearson_p = corr_results['pearson']['p_value']
        
        if pearson_p < 0.001:
            significance = "매우 유의함 (p < 0.001)"
        elif pearson_p < 0.01:
            significance = "매우 유의함 (p < 0.01)"
        elif pearson_p < 0.05:
            significance = "유의함 (p < 0.05)"
        else:
            significance = "유의하지 않음 (p >= 0.05)"
        
        print(f"통계적 유의성: {significance}")
        
        if abs(pearson_corr) >= 0.8:
            strength = "매우 강한"
        elif abs(pearson_corr) >= 0.6:
            strength = "강한"
        elif abs(pearson_corr) >= 0.4:
            strength = "중간 정도의"
        elif abs(pearson_corr) >= 0.2:
            strength = "약한"
        else:
            strength = "매우 약한"
        
        direction = "양의" if pearson_corr > 0 else "음의"
        print(f"상관관계 강도: {strength} {direction} 상관관계")
        
        print("=" * 60)
    
    def run_full_analysis(self):
        """전체 분석 실행"""
        self.print_summary()
        self.plot_scatter()
        self.plot_heatmap()
        self.plot_distribution()

# 사용 예시 함수
def analyze_word_frequency_correlation(df, col1, col2):
    """
    단어 빈도수 컬럼 간 상관관계 분석을 실행하는 편의 함수
    
    Parameters:
    df: pandas DataFrame
    col1: 첫 번째 빈도수 컬럼명
    col2: 두 번째 빈도수 컬럼명
    """
    analyzer = FrequencyCorrelationAnalyzer(df, col1, col2)
    analyzer.run_full_analysis()
    return analyzer

# 예시 사용법
if __name__ == "__main__":
    # 예시 데이터 생성 (실제 데이터로 교체하세요)
    np.random.seed(42)
    n_samples = 1000
    
    # 두 개의 단어 빈도수 컬럼 생성 (상관관계가 있는 데이터)
    freq1 = np.random.poisson(lam=10, size=n_samples)
    freq2 = freq1 * 0.7 + np.random.normal(0, 2, n_samples)  # 상관관계 있는 데이터
    
    example_df = pd.DataFrame({
        'word_freq_corpus1': freq1,
        'word_freq_corpus2': freq2
    })
    
    print("예시 데이터로 상관관계 분석 실행:")
    analyzer = analyze_word_frequency_correlation(example_df, 'word_freq_corpus1', 'word_freq_corpus2') 