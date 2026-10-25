# AI labs

기초인공지능 수업의 Python 구현과 Jupyter 노트북.

| 폴더 | 내용 |
| --- | --- |
| [hw1](hw1) | 선형 회귀, Ridge, Lasso, Elastic Net |
| [hw2](hw2) | Random Forest, AdaBoost, 분류 지표 |
| [hw3](hw3) | 다층 신경망, 활성화 함수, Dropout, Momentum |

## Run

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

노트북은 같은 폴더의 `projectN.py`를 불러옵니다. `hw1` 실행에는 별도의 `crime-train.csv`, `crime-test.csv`가 필요합니다. `hw2`, `hw3`은 scikit-learn으로 실험 데이터를 생성합니다.

`hw3/project3_submitted.py`는 제출 당시의 다른 버전입니다. 노트북은 `project3.py`를 사용합니다.

수업 제공 템플릿을 기반으로 작성했습니다.
