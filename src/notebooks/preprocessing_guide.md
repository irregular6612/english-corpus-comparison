# 빈도수 데이터 전처리 가이드

## 🎯 핵심 질문: Standardization/Normalization이 중요한가?

**답: 네, 매우 중요합니다!** 특히 단어 빈도수 데이터에서는 필수적입니다.

## 📊 빈도수 데이터의 특성

### 문제점들:
1. **극단적 스케일 차이**: "the" (수만 번) vs "rare_word" (1번)
2. **편향된 분포**: 대부분 저빈도, 소수만 고빈도
3. **이상치 영향**: 고빈도 단어가 상관계수에 과도한 영향
4. **비선형 관계**: 로그 스케일에서 선형적일 수 있음

## 🔧 전처리 방법별 특징

### 1. **로그 변환 (Log Transform)**
```python
# log(1+x) 변환 - 가장 권장
log_freq = np.log1p(frequency)
```

**장점:**
- 빈도수 데이터의 표준 전처리
- 극단적 값들의 영향 완화
- 비선형 관계를 선형으로 변환
- 해석이 용이

**사용 시나리오:**
- 대부분의 빈도수 데이터
- 고빈도 단어가 많은 경우
- 상대적 관계가 중요한 경우

### 2. **순위 변환 (Rank Transform)**
```python
# 순위로 변환
rank_freq = frequency.rank()
```

**장점:**
- 이상치에 완전히 강함
- 비선형 관계도 포착
- 분포에 무관하게 동작

**사용 시나리오:**
- 이상치가 많은 경우
- 절대값보다 순서가 중요한 경우
- 스피어만/켄달 상관계수와 유사

### 3. **제곱근 변환 (Square Root Transform)**
```python
# 제곱근 변환
sqrt_freq = np.sqrt(frequency)
```

**장점:**
- 로그보다 덜 극단적
- 중간 정도의 스케일 조정

**사용 시나리오:**
- 로그 변환이 너무 강한 경우
- 중간 정도의 스케일 조정이 필요한 경우

### 4. **표준화 (Standardization)**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
std_freq = scaler.fit_transform(frequency.reshape(-1, 1))
```

**장점:**
- 평균 0, 표준편차 1로 정규화
- 피어슨 상관계수와 호환

**사용 시나리오:**
- 정규분포 가정이 합리적인 경우
- 절대 스케일이 중요한 경우

### 5. **정규화 (Normalization)**
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
norm_freq = scaler.fit_transform(frequency.reshape(-1, 1))
```

**장점:**
- 0~1 범위로 정규화
- 해석이 직관적

**사용 시나리오:**
- 상대적 비율이 중요한 경우
- 0~1 범위가 필요한 경우

### 6. **로버스트 스케일링 (Robust Scaling)**
```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
robust_freq = scaler.fit_transform(frequency.reshape(-1, 1))
```

**장점:**
- 이상치에 강함
- 중앙값과 사분위수 기반

**사용 시나리오:**
- 이상치가 많은 경우
- 중앙값 기반 스케일링이 필요한 경우

## 🎯 권장 전처리 전략

### 기본 전략 (대부분의 경우):
```python
# 1단계: 로그 변환
log_freq1 = np.log1p(df['freq1'])
log_freq2 = np.log1p(df['freq2'])

# 2단계: 상관관계 분석
correlation = stats.pearsonr(log_freq1, log_freq2)
```

### 고급 전략 (이상치가 많은 경우):
```python
# 1단계: 순위 변환
rank_freq1 = df['freq1'].rank()
rank_freq2 = df['freq2'].rank()

# 2단계: 스피어만 상관계수
correlation = stats.spearmanr(rank_freq1, rank_freq2)
```

### 비교 분석 전략:
```python
# 여러 방법을 비교하여 최적 선택
methods = {
    'original': (df['freq1'], df['freq2']),
    'log': (np.log1p(df['freq1']), np.log1p(df['freq2'])),
    'rank': (df['freq1'].rank(), df['freq2'].rank()),
    'robust': (robust_scaler.fit_transform(df[['freq1']]), 
               robust_scaler.fit_transform(df[['freq2']]))
}
```

## 📈 실제 예시 비교

### 시나리오: 수능 영어 vs 교과서 영어 빈도수 비교

```python
# 원본 데이터 (문제가 있는 경우)
original_corr = 0.85  # 고빈도 단어에 의해 과대 추정

# 로그 변환 후
log_corr = 0.72  # 더 현실적인 상관관계

# 순위 변환 후
rank_corr = 0.68  # 순서 기반 상관관계
```

## ⚠️ 주의사항

### 1. **데이터 특성 고려**
- 빈도수 분포 확인 필수
- 이상치 존재 여부 체크
- 스케일 차이 정도 파악

### 2. **해석의 일관성**
- 전처리 방법에 따른 해석 차이
- 원본 스케일 vs 변환된 스케일
- 상대적 관계 vs 절대적 관계

### 3. **검증의 중요성**
- 여러 방법 비교
- 시각적 확인
- 도메인 지식과의 일치성

## 🚀 실무 권장사항

### 1. **기본 워크플로우**
```python
# 1. 데이터 탐색
print(df.describe())
print(df.hist())

# 2. 로그 변환 시도
log_df = np.log1p(df)

# 3. 상관관계 분석
corr = stats.pearsonr(log_df['col1'], log_df['col2'])

# 4. 시각화로 확인
plt.scatter(log_df['col1'], log_df['col2'])
```

### 2. **고급 워크플로우**
```python
# 1. 여러 전처리 방법 비교
from frequency_preprocessing_analysis import analyze_frequency_preprocessing

methods, results = analyze_frequency_preprocessing(df, 'col1', 'col2')

# 2. 최적 방법 선택
best_method = 'log_transform'  # 결과에 따라 선택

# 3. 최종 분석
final_analysis = analyze_word_frequency_correlation(
    preprocessed_df, 'col1_processed', 'col2_processed'
)
```

## 📚 참고 자료

- Zipf's Law: 단어 빈도수 분포의 기본 법칙
- Log-normal Distribution: 빈도수 데이터의 일반적 분포
- Robust Statistics: 이상치에 강한 통계 방법
- Correlation Analysis: 상관관계 분석의 기초

---

**결론: 빈도수 데이터에서는 전처리가 필수이며, 로그 변환이 가장 일반적이고 효과적인 방법입니다.** 