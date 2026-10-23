# 기초인공지능 실습

본인의 수업 제출 자료에서 복원한 Python 구현과 Jupyter 노트북입니다. 수업 제공 구조·문제 설명과 직접 채운 구현이 함께 들어 있습니다. 원래 템플릿과 과제 출처를 개인 창작으로 재표기하지 않았습니다.

| 폴더 | 주제 |
|---|---|
| `hw1` | 선형 회귀, Ridge, Lasso, Elastic Net, 데이터 전처리 |
| `hw2` | Random Forest, AdaBoost, 분류 지표 |
| `hw3` | 다층 신경망, 활성화 함수, Dropout, 경사하강·Momentum |

## 실행

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

각 노트북은 같은 폴더의 `projectN.py`를 사용하므로 해당 폴더를 작업 디렉터리로 열어 실행하세요. `hw1`은 별도 제공 데이터인 `crime-train.csv`, `crime-test.csv`가 필요합니다. 이 데이터는 저장소에 포함하지 않았습니다. `hw2`와 `hw3`의 실험에는 scikit-learn의 데이터 생성 도구가 사용됩니다.

## 복원 상태

노트북의 저장 출력과 실행 횟수, 개인 환경 메타데이터를 제거했습니다. 공개본에 채점 결과·새 학습 결과를 만들어 넣지 않았습니다. 2026-10-24에 Python 소스 및 Python 노트북 셀 구문을 점검했습니다. 모든 학습 실험과 정확도를 새 환경에서 재현한 상태는 아닙니다. 의존성은 기능별 목록이며 과거 제출 환경을 고정한 lockfile이 아닙니다.

새로 작성한 완성형 머신러닝 라이브러리보다는 과거 실습을 살펴보는 학습 기록으로 사용하세요. 수업 제공 코드의 기존 설명과 저작권 표기를 유지합니다.
