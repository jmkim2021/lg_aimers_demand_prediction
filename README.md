# LG Aimers 7기 · 식음업장 메뉴 수요 예측

**메뉴별 최근 28일 판매량으로 향후 7일 판매량을 예측하는 개인 프로젝트의 학습·추론 코드입니다.** 데이터 준비부터 검증, 제출 파일 생성까지 하나의 노트북에 담았습니다.

## 문제와 접근

리조트 식음업장의 메뉴별 수요는 판매량이 0인 날이 많고, 일부 날짜에는 판매량이 크게 늘어납니다. 메뉴마다 요일 패턴도 다릅니다. 한 모델에서 다음 날을 반복 예측하면 앞선 예측의 오류가 뒤로 전파될 수 있어, 예측일 1~7일을 각각 학습하는 방식으로 구성했습니다.

## 구현한 과정

1. 메뉴별 시계열을 28일 입력·7일 정답의 슬라이딩 윈도우로 만들고, 각 윈도우에서 예측일별 학습 샘플을 생성했습니다.
2. 직전 1~7일 판매량, 28일 통계, 최근 7일과 이전 7일의 변화율, 추세 기울기를 피처로 만들었습니다. 예측일의 요일·주말 여부와 같은 요일의 과거 판매량도 추가했습니다.
3. 판매량의 0 빈발과 양수 꼬리를 고려해 Tweedie 목적함수(`variance_power=1.5`)를 사용하는 LightGBM 모델 7개를 학습했습니다. 판매량 0인 샘플의 가중치는 0.2, 양수 샘플은 1.0으로 설정했습니다.
4. 날짜를 기준으로 과거 학습 구간과 최근 30일 검증 구간을 나눴습니다. 검증용 SMAPE는 방향 확인에 사용했으며 대회 평가값과 동일하다고 보지는 않았습니다.
5. 최종 예측을 모델 60%, 전주 같은 요일 판매량 30%, 28일 내 같은 요일 평균 10%로 결합했습니다. 희소 구간에서 단순 기준선도 함께 반영하기 위한 선택입니다.

## 저장소 구성과 실행 조건

- `lg_aimers_demand_prediction.ipynb`: 피처 생성, 7개 모델 학습, 검증, 추론
- `requirements.txt`: 실행 의존성
- `submission_lgbm_20250821_221153.csv`: 보관한 제출 결과

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook lg_aimers_demand_prediction.ipynb
```

대회 데이터는 배포 조건 때문에 저장소에 포함하지 않았습니다. 재실행하려면 노트북 상단의 `BASE_DIR`을 로컬 데이터 위치로 바꿔야 합니다. 제출 파일만으로는 대회 공식 점수를 확인할 수 없어 README에 순위나 점수를 기재하지 않았습니다.

## 사용 예시

`BASE_DIR` 아래에 `train/train.csv`, `test/TEST_*.csv`, `sample_submission.csv`를 두고 노트북 셀을 순서대로 실행합니다. 노트북은 메뉴별 28일 판매 기록을 읽어 7일 예측을 채운 제출 CSV를 `BASE_DIR/lg_aimers_demand_prediction/`에 저장합니다.

## 실행 중 문제가 생기면

- `FileNotFoundError`가 나면 노트북의 `BASE_DIR`과 위 세 데이터 경로를 확인합니다.
- LightGBM 설치가 실패하면 사용 중인 Python 환경에 `requirements.txt`가 설치됐는지 확인합니다.

## 문의·라이선스·변경 이력

질문이나 오류는 [GitHub Issues](https://github.com/jmkim2021/lg_aimers_demand_prediction/issues)에 남길 수 있습니다. 이 저장소에는 별도 코드 라이선스가 명시되어 있지 않으며, 대회 데이터도 포함하지 않았습니다. 변경 내역은 [커밋 기록](https://github.com/jmkim2021/lg_aimers_demand_prediction/commits/main)에서 확인할 수 있습니다.
