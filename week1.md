
# Week 1 summary 

### 키워드 
- EDA
- Data Preprocessing 
- Fature Engineering 
- Data Wrangling 


### 개념 정리 
- EDA (탐색적 데이터 분석)
  - 본격적인 분석에 앞서, 데이터의 특성을 파악하기 위한 과정
  - 통계치, 시각화 활용 
- Data Preprocessing (데이터 전처리)
  - 데이터 품질 문제 처리: 중복값, 결측치, 부정확한 데이터, 이상치 등 
  - 데이터 구조 문제 처리: tidy 조건을 만족할 수 있도록 데이터 정제
    * tidy data 조건: 1) 한 변수는 한 열에, 2) 한 관측치는 한 행에, 3) 하나의 관측 단위는 한 표 안에 들어갈 수 있도록 구성   
  - 데이터 분포, 단위 등 변환 포함 
- Feature Engineering 
   - 주어진 변수들을 재조합하여 분석에 용이한(의미있는) 변수(feature)로 만드는 과정  
- Data Wrangling 
  - 데이터를 보다 쉽게 접근하고 분석할 수 있도록 데이터를 정리하고 통합하는 과정 
  - 데이터 수집, 결합, 정리 과정으로 수행 


### 관련 코드 정리 
- `.shape`: 데이터프레임의 행과 열 개수 파악
- `.info()`: 데이터의 개괄적인 특징 파악 (column 이름, 결측치 여부, 데이터 타입)
- `.describe()`: 숫자형 데이터 통계요약치 제시 
  - `.describe(include="object)`: 범주형 데이터 통계요약치 제시
- `.isnull().sum()`: 데이터 결측치 수 계산 
  - `.fillna()`: 결측치 채우기 
  - `.dropna()`: 결측치 제거 
- `.sort_values(by='변수' )`: 변수의 값을 순서대로 정렬 
- `.duplicated().sum()`: 중복 데이터 수 계산 
- `.set_index()`: 인덱스 설정
- `.reset_index(drop=True)`: 인덱스 재정렬 
- `.groupby()`: 특정 column을 기준으로 정렬해주는 함수
- `.query()`: 데이터프레임에서 필요한 부분 필터링 할 때 사용
  - `df[df["변수명"] == (조건)]`도 사용 가능  
- `.merge()`: 
  - 예: `df1 = df2.merge(df1, how="inner", on=["변수명"])` df1과 df2의 공통 "변수"를 기준으로 교집합(inner)의 형태로 합치기 
- `.concat()`: 데이터 합치기(axis=0이 default)
- `.join()`: 데이터 합치기 


### 좀 더 찾아봐야 하는 내용
- [ ] `.stack()` <-> `.unstack()`
- [ ] `.melt()` <-> `.pivot_table()`
- [ ] 정규표현식 

