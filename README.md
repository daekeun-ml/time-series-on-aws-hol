# Time series on AWS Hands-on Lab

## Introduction
​
이 핸즈온을 통해 여러분은 시계열 데이터(time-series)를 GluonTS와 Amazon SageMaker로 모델링 및 예측하는 방법을 익히게 됩니다.

### 시작하기 전
이 핸즈온은 데이터 과학자나 개발자를 대상으로 한 핸즈온이기에, 머신 러닝에 대한 지식 및 경험이 부족하시다면 Amazon Forecast 핸즈온을 먼저 수행하시는 것을 추천드립니다. 아래와 동일한 데이터셋들에 대한 핸즈온을 수행합니다.

**[Amazon Forecast 핸즈온 바로 가기(한국어)](https://github.com/gonsoomoon-ml/Forecast)** 


### GluonTS란?
Gluon Time Series(GluonTS)는 MXNet 프레임워크에 기반한 Gluon Toolkit입니다.
Baseline 수립을 위한 기본적인 알고리즘부터 딥러닝 기반 확률적 시계열 모델링이 가능한 알고리즘까지 다양한 빌트인 알고리즘들이 내장되어 있으며, 또한 예측 결과를 쉽게 평하하고 비교할 수 있는
기능들도 포함되어 있습니다.

GluonTS supported algorithms:
- Deep Factor: https://arxiv.org/abs/1905.12417
- DeepAR: https://arxiv.org/abs/1704.04110
- DeepState: https://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting.pdf
- DeepVAR(Multivariate variant of DeepAR):  https://arxiv.org/abs/1910.03002
- Gaussian Processes Forecaster
- Low-Rank Gassian Copula Processes (GPVAR):  https://arxiv.org/abs/1910.03002
- LSTNet: https://arxiv.org/abs/1703.07015
- Non-Parametric Time Series Forecaster (NPTS)
- Prophet: https://facebook.github.io/prophet/
- Seasonal Naive
- Seq2Seq
- Simple Feedforward (MLP)
- Transformer: https://arxiv.org/pdf/1706.03762.pdf
- Wavenet: https://arxiv.org/pdf/1609.03499.pdf


## Hands-on Lab 
사용 사례에 따라 3가지 데이터셋에 대해 실습해 볼 수 있습니다.<br>
만약 시계열 데이터를 다뤄본 경험이 없다면 1번부터 시작하는 것을 권장하며, 어느 정도 시계열 데이터를 다루는 데 익숙하다면 1,2번을 건너뛰고 3번만 진행하시면 됩니다.

### 1. Store Item Demand (Beginner)
본 데이터셋은 10개의 상점, 50개의 아이템으로 **일단위** 판매 수량을 예측하는 데이터셋입니다.
별도의 아이템 메타데이터나 연관 시계열 데이터가 없는 가장 간단한 데이터로, `related time series` 및 `item metadata` 없이 `target time series`만으로 훈련 및 예측을 수행하게 됩니다.

- **[바로 가기](store-item-demand/)**    
- Dataset description: https://www.kaggle.com/c/demand-forecasting-kernels-only/overview 

### 2. Traffic Volume 예측 (Intermediate)

본 데이터셋은 미국 미네아폴리스 서쪽 방향 고속도로(I-94)의 2012년~2018년 차량 통행량을 **시간 단위**로 예측하는 데이터셋입니다. store-item-demand 데이터셋과 달리 결측 시계열들이 있고 연관 시계열 데이터도 존재하기 때문에 간단한 데이터 전처리를 수행 후에 `target time series` 와 `related time series` 로 훈련 및 예측을 수행하겠습니다.

- **[바로 가기](traffic-volume/)**    
- Dataset description: https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume


### 3. Walmart Store Sales (Intermediate~Advanced)

본 데이터셋은 2010-02-05에서 2012-11-01까지의 시계열 데이터로 45개의 월마트 매장의 **주단위** 매출액을 예측하는 데이터셋입니다. 각 매장에서는 여러 부서가 있으며, 각 상점의 부서 전체 매출을 예측하는 문제입니다.
대상 시계열, 연관 시계열 데이터, 아이템 메타데이터를 모두 사용하기 때문에  `target time series`, `related time series`, `item metadata` 를 모두 활용하여 훈련 및 예측을 수행하게 됩니다.

- **[바로 가기](walmart-sale/)**    
- Dataset description: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting


## References
- GluonTS Paper: https://arxiv.org/abs/1906.05264
- GluonTS 핸즈온: https://gluonts-workshop.go-aws.com/
- Amazon Forecast 핸즈온: https://github.com/chrisking/ForecastPOC
- Amazon Forecast 핸즈온(한국어): https://github.com/gonsoomoon-ml/Forecast/tree/master/StoreItemDemand
- Amazon SageMaker 개발자 가이드: https://aws.amazon.com/sagemaker/developer-resources/


## License Summary

이 샘플 코드는 MIT-0 라이센스에 따라 제공됩니다. LICENSE 파일을 참조하십시오.