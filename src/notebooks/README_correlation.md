# 단어 빈도수 상관관계 분석 가이드

이 프로젝트는 두 개의 단어 빈도수 컬럼 간의 상관관계를 분석하는 도구입니다.
notion link: https://obsidian-april-566.notion.site/Corpus-1b5c0cefb4a28030b7fee47d684ca216?source=copy_link

## 📋 기능

- **다양한 상관계수 계산**: 피어슨, 스피어만, 켄달 타우
- **시각화**: 산점도, 히트맵, 분포도, 박스플롯
- **통계적 유의성 검정**: p-value 계산
- **이상치 탐지**: 상관관계에서 벗어나는 단어들 식별
- **한글 지원**: 결과 출력 및 그래프에 한글 표시

## 🚀 빠른 시작

### 1. 기본 사용법

```python
import pandas as pd
from correlation_analysis import analyze_word_frequency_correlation

# 데이터 로드
df = pd.read_csv('your_data.csv')

# 상관관계 분석 실행
analyzer = analyze_word_frequency_correlation(df, 'freq_column1', 'freq_column2')
```

### 2. 클래스 직접 사용

```python
from correlation_analysis import FrequencyCorrelationAnalyzer

# 분석기 생성
analyzer = FrequencyCorrelationAnalyzer(df, 'freq_column1', 'freq_column2')

# 개별 분석 실행
analyzer.print_summary()      # 통계 요약
analyzer.plot_scatter()       # 산점도
analyzer.plot_heatmap()       # 히트맵
analyzer.plot_distribution()  # 분포도
```

## 📊 분석 결과 해석

### 상관계수 해석
- **0.8 이상**: 매우 강한 상관관계
- **0.6-0.8**: 강한 상관관계
- **0.4-0.6**: 중간 정도의 상관관계
- **0.2-0.4**: 약한 상관관계
- **0.2 미만**: 매우 약한 상관관계

### 통계적 유의성
- **p < 0.001**: 매우 유의함
- **p < 0.01**: 매우 유의함
- **p < 0.05**: 유의함
- **p ≥ 0.05**: 유의하지 않음

## 🔍 상관계수 종류

### 1. 피어슨 상관계수 (Pearson)
- **용도**: 선형 관계 측정
- **범위**: -1 ~ +1
- **특징**: 정규분포 가정, 이상치에 민감

### 2. 스피어만 상관계수 (Spearman)
- **용도**: 순위 관계 측정
- **범위**: -1 ~ +1
- **특징**: 비선형 관계도 측정 가능, 이상치에 덜 민감

### 3. 켄달 타우 (Kendall's Tau)
- **용도**: 순위 관계 측정
- **범위**: -1 ~ +1
- **특징**: 스피어만보다 이상치에 더 강함

## 📈 시각화 종류

### 1. 산점도 (Scatter Plot)
- 원본 데이터와 로그 변환 데이터 비교
- 상관관계 패턴 시각화

### 2. 히트맵 (Heatmap)
- 상관계수 행렬 시각화
- 색상으로 상관관계 강도 표현

### 3. 분포도 (Distribution)
- 히스토그램과 박스플롯
- 데이터 분포 특성 파악

## 🎯 실제 사용 예시

### 예시 1: 두 코퍼스 간 단어 빈도 비교
```python
# 수능 영어와 교과서 영어 코퍼스 비교
df = pd.read_excel('corpus_comparison.xlsx')
analyzer = analyze_word_frequency_correlation(df, '수능_빈도', '교과서_빈도')
```

### 예시 2: 상위 빈도 단어만 분석
```python
# 상위 100개 단어만 선택
top_words = df.nlargest(100, 'freq_column1')
analyzer = analyze_word_frequency_correlation(top_words, 'freq_column1', 'freq_column2')
```

### 예시 3: 이상치 단어 찾기
```python
# 상관관계에서 벗어나는 단어들 식별
corr_results = analyzer.calculate_correlation()
# 잔차 분석을 통해 이상치 탐지
```

## ⚠️ 주의사항

1. **결측값 처리**: 자동으로 결측값이 제거됩니다
2. **데이터 타입**: 빈도수는 숫자형이어야 합니다
3. **한글 폰트**: macOS에서는 'AppleGothic' 사용
4. **메모리**: 대용량 데이터의 경우 메모리 사용량에 주의

## 📦 필요한 라이브러리

```bash
pip install pandas numpy matplotlib seaborn scipy
```

또는 `uv` 사용:
```bash
uv add pandas numpy matplotlib seaborn scipy
```

## 🔧 커스터마이징

### 그래프 스타일 변경
```python
# 그래프 크기 조정
analyzer.plot_scatter(figsize=(12, 10))

# 색상 테마 변경
plt.style.use('seaborn-v0_8')
```

### 추가 분석
```python
# 상관계수만 계산
corr_results = analyzer.calculate_correlation()
print(corr_results['pearson']['correlation'])

# 특정 시각화만 실행
analyzer.plot_scatter()
```

## 📞 문의사항

코드 사용 중 문제가 있거나 개선 사항이 있으면 이슈를 등록해주세요. 