# LG Aimers 7기 · 식음업장 메뉴 수요 예측

리조트 내 식음업장의 **28일 판매 이력으로부터 향후 7일 메뉴별 판매량을 예측**하는 과제 솔루션입니다.
판매량 분포에 0이 매우 많고(희소) 양의 꼬리가 길며 메뉴별 패턴 편차가 크다는 특성을 고려해
**Tweedie LightGBM + Direct 멀티-호라이즌 + Naive 시계열 블렌딩** 구조로 SMAPE 안정성을 극대화했습니다.

## 결과

- 베스트 제출: [`submission_lgbm_20250821_221153.csv`](./submission_lgbm_20250821_221153.csv)
- 평가지표: 업장 동일가중 SMAPE

## 파이프라인

```
[28일 메뉴별 시퀀스]
      │
      ├──▶ window_features      (lag1~7, 평균/표준편차/추세 기울기, 추세비)
      ├──▶ horizon_context(h)   (예측 시점 요일·주말 플래그, 동요일 평균/표준편차, lag7)
      │
      ▼
[Tweedie LGBMRegressor × 7]   ← Horizon별 Direct 학습 (비재귀)
      │
      ▼
[모델 / lag7(naive seasonal) / 동요일 평균  가중 블렌딩]
      W_MODEL=0.6 · W_LAG7=0.3 · W_MEAN=0.1
      │
      ▼
[zero floor 1e-6] → submission.csv
```

### 1. Direct 멀티-호라이즌 슬라이딩 윈도우 데이터셋
메뉴별 시계열을 *28일 입력 → 7일 예측* 단위 슬라이딩 윈도우로 변환하고, 각 윈도우에 호라이즌(`h=1..7`)을 결합해
한 시점에서 7개의 학습 샘플을 만드는 구조를 설계했습니다. 메뉴명을 정수 `menu_id`로 인코딩해 트리 모델 입력에 통합했습니다.

### 2. 윈도우 통계 피처 엔지니어링
직전 1~7일 lag, 28일 평균·표준편차·최솟값·최댓값·합계, 최근 7일 평균 대비 직전 7일 평균 비율(`trend_7_7`),
28일 선형회귀 기울기(`win_slope`) 등으로 **단기 변동성과 중기 추세를 동시에** 표현합니다.

### 3. 호라이즌별 컨텍스트 피처 (요일·주말 패턴 학습)
`datetime`을 분해해 예측 시점(`T+h`)의 요일(`dow_h`)과 주말 플래그(`is_weekend_h`)를 생성하고,
28일 윈도우 내 *동일 요일 평균·표준편차*, *전주 동요일 실측값(`lag7_for_h`)*, `lag7 / dow-mean` 비율을 추가하여
**요일 효과·주간 계절성**을 모델이 인지하도록 했습니다.

### 4. Tweedie LGBM × 7 (Horizon별 Direct 학습)
판매량의 *0 빈발 + 양수 꼬리* 분포에 맞춰 `objective='tweedie', tweedie_variance_power=1.5`를 채택했고,
**호라이즌마다 독립적인 LGBMRegressor 7개**를 학습해 재귀예측 오차 누적을 차단했습니다.
(`n_estimators=1200, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8`)

### 5. SMAPE 정렬 샘플 가중
`y=0` 샘플에 **0.2**, `y>0` 샘플에 **1.0** 가중치를 부여하여 영구간이 손실을 지배하는 과적합을 억제하고
SMAPE 분모 구조와 학습 목적을 정렬했습니다.

### 6. 시간 기반 학습/검증 분할
윈도우의 `last_date` 기준으로 *학습 = `global_last − 30일` 이전*, *검증 = 최근 30일* 으로 분할해
**데이터 누설을 차단**하고 실 운영 시나리오와 동일한 평가 환경을 구성했습니다.

### 7. 모델·Naive 시계열 하이브리드 블렌딩
최종 예측은 *(1) LGBM 모델 예측*, *(2) 전주 동요일 실측값(naive seasonal, lag7)*, *(3) 28일 내 동요일 평균* 을
**0.6 : 0.3 : 0.1** 고정 가중으로 결합합니다. 강건한 단순 baseline을 30% 비중으로 항상 섞어 **희소·이상 구간의 분산을 흡수**했고,
`max(ŷ, 1e-6)` floor로 **SMAPE 분모 0**을 방지했습니다.

## 실행 방법

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook lg_aimers_demand_prediction.ipynb
```

> 데이터 (`train/`, `test/`, `sample_submission.csv`)는 LG Aimers 7기 대회 데이터 라이선스로 인해 본 리포에 포함되지 않습니다.
> 실행 전 노트북 상단의 `BASE_DIR`을 본인 환경에 맞게 수정해 주세요.

## 폴더 구조

```
.
├── lg_aimers_demand_prediction.ipynb   # 학습·추론·제출 생성 (마크다운 문서 포함)
├── submission_lgbm_20250821_221153.csv # 베스트 제출 결과
├── requirements.txt
├── .gitignore
└── README.md
```

## 주요 하이퍼파라미터

| 항목 | 값 |
|---|---|
| `LOOKBACK` / `HORIZON` | 28 / 7 |
| LGBM `objective` | `tweedie` (`variance_power=1.5`) |
| LGBM `n_estimators` / `learning_rate` | 1200 / 0.03 |
| Sample weight (`y=0` / `y>0`) | 0.2 / 1.0 |
| 블렌딩 `W_MODEL / W_LAG7 / W_MEAN` | 0.6 / 0.3 / 0.1 |
| Zero floor `EPS` | 1e-6 |
