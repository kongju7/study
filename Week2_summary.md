# Week 2 Summary 

### 키워드 
- 확률 
- 이항분포  
- 조건부 확률  
- 베이지안 정리 
- 중심 극한 정리  
- 부트스트래핑 기법 
- 신뢰 구간 
- 가설 검정
  - 귀무가설, 대립가설
  - 제1종 오류, 제2종 오류  
- A|B 테스트 



### 개념 정리 
- 확률(Probability): 어떤 사건에서 어떤 일이 일어날 가능성을 수로 나타낸 것  
  
- 이항분포(Binomial Distribution)): 독립적으로 반복되는 행위에 의해서 실패 또는 성공과 깉이 두 가지  가능성만 가지는 사건의 확률을 결정하는 함수  

- 조건부 확률((Conditional Distribution): 어떠한 사건의 결과에 의해 영향을 받는 한 사건의 결과에 대한 확률 
  - `P(A|B) = P(A∩B) / P(B)`  
  - `P(C)`란 암에 걸릴 확률  
  - `P(Pos|C)`란 암에 걸린 환자 중 테스트를 통해 암에 걸렸다고 나올 확률  
  - `P(Neg|^C)`란 암에 걸리지 않은 환자 중 테스트를 통헤 암에 걸리지 않았다고 나올 확률  
  
- 베이지안 정리  
  - `P(A|B) = P(A∩B) / P(B) = P(B|A)P(A) / P(B)`
  - 사전 확률(prior probability)에 우리가 확인한 증거(data)를 사후 확률(posterior probability)에 포함하는 것  
  - Posterior Probability(사후확률)  = Prior Probability(사전 확률) x Evidence(데이터를 통해 확인한 증거)  

- 중심 극한 정리(Central Limit Theorem): 모집단의 분포에 사오간없이 임의의 분포에서 추출된 표본들의 평균 분포는 정규분포를 이룸. 
  - 통상적으로 표본의 수가 30개를 넘어갈 때 적용 가능
 
- 부트스트래핑(Bootstrapping) 기법: 우리가 가지고 있는 표본을 모집단으로 가정하고 중복 추출을 허용하여 원하는 개수의 데이터를 추출하는 기법   
  - 표본 평균 분포를 얻을 수 있음   

- 신뢰 구간(Confidence Interval): 모수를 포함하고 있을 구간을 확률과 함께 제시. 모수가 포함되어 있을 구간을 확률과 함께 제공하여 불확실성을 줄이고, 우리가 추론한 모수의 신뢰성을 가늠할 때 사용  

- 가설 검정(Hypothesis Test): 우리가 알고자 하는 모집단의 특성에 대한 가설을 세우고, 이 가설을 표본을 활용하여 추론하는 과정  
  - 귀무가설(null hypothesis): 표본을 수집하기 전 사실이라고 믿는 가설. 가설 검정의 대상이 됨.(수학적으로 = 을 포함함)  
  - 대립가설(alternative hypothesis): 귀무가설과 대립되는 가설로, 우리가 사실이라고 증명하고자 하는 가설   
  - 제1종 오류: 귀무가설이 참임에도 불구하고 기각하는 오류. False Positive  
  - 제2종 오류: 귀무가설이 거짓임에도 불구하고 기각하지 않는 오류. False Negative  

- A|B 테스트: 모든 조건이 동일한 상황에서 대조군과 실험군에게 특정 조건 하나만 달리하여 결과의 차이를 확인하는 테스트
  - 웹 사이트 디자인이나 이메일 마케팅 캠페인, 광고 및 콘텐츠 전략 등에 활용되고 있음



### 관련 코드 정리 
- `np.random.seed(42)`: 난수 유형 고정 
- `np.random.choice(데이터, 한 번에 뽑을 데이터 수, replace=True)`
- `데이터.sample(한 번에 뽑을 데이터 수, replace=True)`
- `np.mean()` : 리스트의 평균 값 계산 
- `low, upper = np.percentile(means, 2.5),np.percentile(means, 97.5)`: 신뢰구간 구하기
- `np.random.normal(기준 값, sample_std, 10000) `: 귀무가설에 기반한 분포 그리기 
- `(null_vals > sample_mean).mean()`: 귀무가설에 기반한 분포에서 p-value 확인 
- `데이터.str.replace(',', '')`: 문자 중간에 포함되어 있는 쉼표 삭제 
- `stats.ttest_ind(데이터A, 데이터B, alternative="greater")`: 두 집단의 평균 차이를 검정하기 위한 t-test (alternative은 대립가설을 기준으로 표기)
- `z_score, p_value = sm.stats.proportions_ztest([대조군 결과, 실험군 결과], [대조군 수, 실험군 수], alternative='smaller')`: 두 집단의 평균 차이를 검정하기 위한 z-tset



### 좀 더 찾아봐야 하는 내용
- [ ] 베이지안 정리 
- [ ] 부트스트래핑 기법 
- [ ] A|B 테스트 활용 사례 
