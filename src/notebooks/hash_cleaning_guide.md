# '#' 값 정리 가이드

데이터프레임에서 '#'이 들어간 셀을 None 값으로 변경하는 방법을 설명합니다.

## 🚀 빠른 시작

### 1. 기본 사용법
```python
import pandas as pd
from clean_hash_values import clean_hash_values

# 데이터 로드
df = pd.read_csv('your_data.csv')

# 모든 컬럼에서 '#' 값을 None으로 변경
cleaned_df = clean_hash_values(df)
```

### 2. 특정 컬럼만 정리
```python
# 특정 컬럼만 처리
cleaned_df = clean_hash_values(df, columns=['name', 'age', 'city'])
```

### 3. 상세 보고서와 함께 정리
```python
from clean_hash_values import clean_hash_values_detailed

# 상세한 보고서와 함께 정리
cleaned_df, report = clean_hash_values_detailed(df)
```

## 📊 함수별 특징

### 1. `clean_hash_values()` - 기본 함수
- 모든 컬럼에서 '#' 값을 None으로 변경
- 간단하고 빠른 처리

### 2. `clean_hash_values_detailed()` - 상세 보고서
- 처리 결과에 대한 상세한 보고서 제공
- 컬럼별 '#' 개수 통계
- 처리된 컬럼 목록

### 3. `clean_hash_values_advanced()` - 고급 옵션
- 대체값을 자유롭게 설정 가능
- None, 'MISSING', 0 등 원하는 값으로 대체

## 🎯 사용 예시

### 예시 1: 기본 정리
```python
# 원본 데이터
df = pd.DataFrame({
    'name': ['John', '#', 'Jane'],
    'age': [25, '#', 30],
    'city': ['NYC', 'LA', '#']
})

# 정리
cleaned_df = clean_hash_values(df)
print(cleaned_df)
```

### 예시 2: 특정 컬럼만 정리
```python
# name과 city 컬럼만 정리
cleaned_df = clean_hash_values(df, columns=['name', 'city'])
```

### 예시 3: 다른 값으로 대체
```python
from clean_hash_values import clean_hash_values_advanced

# '#'을 'MISSING'으로 대체
cleaned_df = clean_hash_values_advanced(df, replacement_value='MISSING')

# '#'을 0으로 대체
cleaned_df = clean_hash_values_advanced(df, replacement_value=0)
```

### 예시 4: 상세 보고서
```python
cleaned_df, report = clean_hash_values_detailed(df)

# 보고서 정보 확인
print(f"총 발견된 '#' 개수: {report['total_hash_found']}")
print(f"처리된 컬럼: {report['processed_columns']}")
```

## ⚠️ 주의사항

### 1. 데이터 타입
- 문자열(object) 컬럼만 처리됩니다
- 숫자 컬럼은 자동으로 제외됩니다

### 2. 처리 방식
- 정확히 '#'인 값만 처리
- '#text' 같은 부분 포함 값은 처리하지 않음
- 공백이 포함된 '# '도 처리

### 3. 원본 데이터 보존
- 원본 데이터프레임은 변경되지 않습니다
- 새로운 데이터프레임이 반환됩니다

## 🔧 고급 사용법

### 1. 조건부 처리
```python
# 특정 조건에서만 정리
if df.isin(['#']).any().any():
    cleaned_df = clean_hash_values(df)
    print("'#' 값이 정리되었습니다.")
else:
    print("'#' 값이 없습니다.")
```

### 2. 배치 처리
```python
# 여러 파일 처리
files = ['data1.csv', 'data2.csv', 'data3.csv']

for file in files:
    df = pd.read_csv(file)
    cleaned_df = clean_hash_values(df)
    cleaned_df.to_csv(f'cleaned_{file}', index=False)
```

### 3. 커스텀 대체값
```python
# 컬럼별로 다른 대체값 사용
df['name'] = df['name'].replace('#', 'Unknown')
df['age'] = df['age'].replace('#', -1)
df['city'] = df['city'].replace('#', 'Unknown')
```

## 📈 성능 최적화

### 1. 대용량 데이터 처리
```python
# 청크 단위로 처리
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    cleaned_chunk = clean_hash_values(chunk)
    # 처리된 청크 저장
```

### 2. 메모리 효율성
```python
# 필요한 컬럼만 로드
df = pd.read_csv('data.csv', usecols=['name', 'age', 'city'])
cleaned_df = clean_hash_values(df)
```

## 🐛 문제 해결

### 1. '#' 값이 처리되지 않는 경우
```python
# 데이터 확인
print(df.dtypes)  # 컬럼 타입 확인
print(df.isin(['#']).sum())  # '#' 개수 확인
```

### 2. 부분 일치 처리
```python
# '#text' 같은 값도 처리하려면
df = df.replace(r'^#.*$', None, regex=True)
```

### 3. 공백 포함 값 처리
```python
# '# ' 같은 값도 처리
df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)
cleaned_df = clean_hash_values(df)
```

## 📞 추가 도움

코드 사용 중 문제가 있거나 개선 사항이 있으면 이슈를 등록해주세요. 