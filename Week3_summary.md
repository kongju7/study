## Week 3 Summary 

### 키워드 
- 선형대수(Linear Algebra): 스칼라(scalar), 벡터(vector), 행렬(matrix) 
- 공분산(Covariance)
- 상관계수(Correlation coefficient)
- 주성분 분석(Principal Component Analysis, PCA) 
- 군집화(Clustering) 
- RFM(Recency, Frequency, Monetary)
- 데이터 스케일링(Data Scaling), 피처 스케일링(Feature Scaling)
- 경사하강법(Gradient Descent)


### 개념 정리 
-  스칼라(scalar): 상수 또는 변수로 저장되어 있는 숫자 
  - 성분: 행렬을 구성하고 있는 각각의 수 또는 문자 
  
-  벡터(vector): 성분을 1차원 형태로 배열한 것  
  - 벡터의 차원: 성분의 개수 
  - 벡터의 크기(Norm, Length, Magnitude): 벡터의 길이. 항상 양의 값을 가짐. 피타고라스 정리를 통해서 구할 수 있음.  
  - 벡터의 내적(Dot Product): 두 벡터에 대해서 서로 대응하는 각각의 성분을 곱한 뒤 모두 합하여 구함. (이 때 두 벡터의 차원은 같아야 함.) 
  - 벡터의 직교(Orthogonality): 두 벡터의 내적이 0이면 두 벡터는 서로 수직임. 
  - 단위 벡터(Unit Vector): 길이가 1인 벡터. 모든 벡터는 단위 벡터의 선형 결합(linear combination)으로 표기할 수 있음.  
  
-  행렬(matrix): 성분을 2차원 형태로 배열한 것 
  - 행렬의 차원: 행과 열의 개수 
  - 행렬의 전치(Transpose): 행과 열을 바꾸어 나타내는 것. 전치의 전치는 자기 자신임. 
  - 행렬곱(Matrix Multiplication): 두 행렬에 대해 앞 행렬의 열과 뒷 행렬의 행의 수가 같으면 행렬끼리 곱할 수 있음. 행렬곱의 결과는 행렬임. 
  - 정사각 행렬(Square Matrix): 행과 열의 동일한 행렬 
    - 대각 행렬(Diagonal Matrix): 주 대각선(Principal diagonal)을 제외한 모든 성분이 0인 정사각 행렬
    - 단위 행렬(Identity Matrix): 대각 행렬 중 주 대각선 성분이 모두 1인 행렬. 임의의 정사각 행렬에 단위 행렬을 곱한 것은 자기 자신과 같음.  
    - 역행렬(Inverse Matrix): 임의의 정사각 행렬에 대하여 곱했을 때 단위 행렬이 되도록 하는 행렬
  - 행렬식(Determinant): 행렬식이 0이면 역행렬이 존재하지 않음.   
  
- 스팬(Span): 주어진 두 벡터의 조합으로 만들 수 있는 모든 가능한 벡터의 집합 
  - 선형 관계의 벡터: 두 벡터가 같은 선상에 있는 경우, 이 벡터들은 선형 관계에 있다고 표현함. 선형 관계에 있는 두 벡터는 조합을 통해서 선 외부의 새로운 벡터를 생성할 수 없음. 선형 관계에 있는 벡터들이 만들어내는 Span은 벡터의 수보다 더 적은 차원을 가짐. 
  - 선형 관계가 없는 벡터 = 선형 독립 벡터: 선형 독립 벡터들이 만들어내는 Span의 차원은 벡터의 수와 같음.  
- 기저(Basis): 벡터 공간 V의 기저는 V라는 공간을 채울 수 있는 선형 관계에 있지 않은 벡터들의 모음(Span의 역개념). 선형 독립인 두 벡터는 벡터 공간 R^2(2차원 평면)의 basis
- 랭크(Rank): 행렬의 열을 이루는 벡터들로 만들 수 있는 공간(span)의 차원. 행렬의 선형 독립인 행 또는 열의 최대 개수.  

- 벡터 변환(Vector Transformation): 임의의 벡터는 함수(행렬) f에 의해 변환될 수 있음. 벡터의 변환은 벡터와 행렬 T의 곱으로 이루어짐. 
  - Eigenvector: 주어진 변환에 의해서 크기만 변하고 방향은 변하지 않는 벡터 
  - Eigenvalue: Eigenvector의 변화한 크기 값

- 공분산(Covariance): 두 변수에 대하여 한 변수가 변화할 때 다른 변수가 어떠한 연관성을 갖고 변하는지 나타낸 값. 두 변수의 연관성이 클수록 공분산의 값도 커짐.  
- 상관계수(Correlation coefficient): 공분산을 두 변수의 표준편차로 나눠준 값. -1~1 사이 값을 가짐. 
- 주성분 분석(Principal Component Analysis, PCA): 원래 데이터의 정보(분산)을 최대한 보존하는 새로운 축을 찾고, 그 축에 데이터를 사영(Linear Projection)하여 고차원의 데이터를 저차원으로 변환하는 기법 
  - 주성분(PC): 기존 데이터의 분산을 최대한 보존하도록 데이터를 projection하는 축 
  - PC의 단위벡터는 데이터의 공분산 행렬에 대한 Eigenvector임. 
  - Eigenvector에 projectiong한 데이터의 분산이 Eigenvalue 
  - 특성 추출(feature extraction) 방식 중 하나로, 기존 변수의 선형결합으로 새로운 변수(PC)를 생성함. 
    - 특성 추출(feature extraction): 기존 feature 간 상관관계를 고려하여 조합하여, 새로운 feature를 추출(생성)함. 원래 변수들의 선형결합으로 이루어지며, 해석이 어렵다는 단점이 있으나 feature 수를 큰 폭으로 줄일 수 있음. 
    - 특성 선택(feature selection): 데이터 셋에서 덜 중요한 feature을 제거하는 방법. feature의 해석이 쉽다는 장점이 있으나, feature간 연관성을 직접 고려해야 함. 
 
- 군집화(Clustering): 별도의 레이블이 없는 데이터 안에서 패턴과 구조를 발견하는 비지도 학습의 대표적인 기술. 서로 유사한 데이터는 같은 집단으로 묶고, 서로 유사하지 않는 데이터는 다른 집단으로 분리함. 
  - K-Means Clustering = Centroid Based Clustering Algorithm. 군집 중심성(Centroid)라는 특정 임의의 지점을 선택해 해당 중심으로부터 거리가 가장 가까운 포인트를 같으 집단, 즉 비슷한 특성을 가진 데이터들이 모인 집단으로 묶는 방법. 
    - 거리 계산 방법: Euclidean Distance, Cosine Similarity, Jaccard Distance 등이 활용됨. 
    - Elbow Method: K-Means 군집화 알고리즘의 성능을 최대화시키는 적합한 k의 개수를 선택하는 방법. 다양한 k값을 이용해 데이터를 군집화한 뒤, 각 k값에 해당하는 군집 내의 데이터들이 얼마나 퍼져 있는지 응집도, 즉 inertia를 값으로 확인함. 일반적으로 intertia가 급격하게 변하는 지점을 최적의 k로 설정함.  
    
- RFM(Recency, Frequency, Monetary)
  - Recency: 고객의 마지막 구매 시점을 나타내는 것으로, 산업에 따라 다소 차이가 있지만 일반적으로 최근에 구매한 고객일수록 현재의 관계에 유의하다고 판단함. 
  - Frequency: 구매 빈도로서 고객이 정해진 기간동안 얼마나 자주 구매했는지를 나타내는 지표로, 고객의 구매 또는 이용 활동성 등을 판단함.  
  - Monetary: 일정 기간 동안에 고객의 총 구매 금액을 나타내는 지표로, 지나치게 높은 구매액이 존재할 경우 측정시에 상한선을 두어 전체적인 지수 왜곡을 방지할 수 있음.  

- 데이터 스케일링(Data Scaling), 피처 스케일링(Feature Scaling): 데이터의 단위 차이에 따른 왜곡을 방지하기 위해서 데이터 값의 범위를 유사하게 맞춰주는 과정.  
  - 정규화와 표준화의 방법이 있음.  

- 경사하강법(Gradient Descent): 예측값과 실제값 사이의 차이를 나타내는 오차 함수가 최소값을 갖도록 미분을 활용하여 독립변수를 찾는 방법  
  - 학습률(Learning Rate)
    - 학습률이 너무 낮으면 알고리즘이 수렴하기 위해서 반복을 많이 해야 되어 시간이 오래 소요되고, 반대로 학습률이 너무 크면 극소값을 지나쳐 알고리즘이 수렴을 못하고 계산을 계속 반복하게 되는 일이 일어날 수 있음. 

  
### 관련 코드 정리 
- `np.dot()`: 벡터의 내적 구하기 
- `np.T()` 또는 `np.transpose()`: 전치  
- `np.matmul()`: 행렬곱 
- `np.identity()` 또는 `np.eye()`: 단위 행렬
- `np.linalg.inv()`: 역행렬 
- `np.linalg.det()`: 행렬식
- `np.linalg.eig()`: Eigenstuff 
- `df.cov()` 또는 `np.cov()`: 공분산 
- `df.corr()` 또는 `np.corrcoef()`: 상관관계 
- `StandardScaler()`: 데이터 스케일링 - 표준화 
- `MinMaxScaler()`: 데이터 스케일링 - 정규화 
- `np.log()`: 로그 변환 
- `PCA()`: 주성분 분석 
- `KMeans()`: K-means clustering 
- `math.e`: 무리수 


### 좀 더 찾아봐야 하는 내용
- [ ] Eigenstuff 

