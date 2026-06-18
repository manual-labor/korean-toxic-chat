# 한국어 악성 채팅 탐지 모델

한국어 인터넷 채팅을 정상(normal)과 악성(toxic)으로 분류하는 2라벨 분류 프로젝트이다.
기본 `klue/roberta-small`을 파인 튜닝한 모델과 인터넷 채팅 특화 단어집 및 TAPT(Task-Adaptive Pre-Training)를 적용한 모델을 비교하여, 비정형 채팅 표현에 대한 분류 성능 변화를 확인하였다.

본 저장소는 연구의 실험 코드 전체를 제공한다. 데이터 구축 과정, 알고리즘의 이론적 배경 및 상세 실험 분석은 별도 제출 보고서에 기술하였다.

## 주요 내용

* 기반 모델: `klue/roberta-small`
* 분류 클래스: `normal`, `toxic`
* 최대 입력 길이: 64 tokens
* 텍스트 최대 길이: 64자
* 주요 적용 기법

  * 인터넷 채팅 특화 신규 단어집
  * TAPT 기반 추가 사전학습
  * LLRD(Layer-wise Learning Rate Decay)
  * AdamW 및 학습률 스케줄링
  * Early Stopping
* 평가 지표

  * Accuracy
  * Macro Precision
  * Macro Recall
  * Macro F1-score
  * Toxic class F1-score
  * Confusion Matrix

## 데이터 구성

학습 데이터는 직접 구축한 한국어 인터넷 채팅 데이터로 구성하였다.

* `normal`: 일반적인 대화
* `toxic` : 비하, 혐오, 공격적 표현 및 괴롭힘이 포함된 대화
* `chat_log`: TAPT에 사용하는 비라벨 채팅 말뭉치
* `test_data`: 학습 및 검증 데이터와 분리된 독립 테스트 데이터

전처리 과정에서는 NFKC 정규화, 제로폭 문자 제거, 공백 정리, 중복 제거 및 테스트 데이터와의 중복 제거를 수행하였다. 검증 및 테스트 문장은 TAPT 말뭉치에서도 제외하여 데이터 누수를 방지하였다.

## 디렉터리 구성

```text
.
├── train.py
├── new_words.txt
├── learning_data/
│   ├── toxic.json
│   └── chat_log.json
├── test_data/
│   ├── normal.csv
│   └── toxic.csv
└── outputs_low_spec/
```

`train.py`가 아닌 다른 파일명을 사용하는 경우 실행 명령의 파일명을 해당 이름으로 변경해야 한다.

## 실행 환경

Python 환경에서 다음 라이브러리가 필요하다.

```bash
pip install torch transformers scikit-learn numpy
```

CUDA 사용이 가능한 경우 GPU를 자동으로 사용하며, CUDA를 사용할 수 없는 환경에서는 CPU로 실행된다.

## 실행 방법

### 기본 모델

신규 단어집과 TAPT를 적용하지 않고 `klue/roberta-small`을 직접 파인튜닝한다.

```bash
python train.py --tapt off --new-words off
```

### 제안 모델

신규 단어집을 적용하고 인터넷 채팅 말뭉치로 TAPT를 수행한 후 분류 모델을 학습한다.

```bash
python train.py --tapt on --new-words on
```

기존 TAPT 모델 캐시를 사용하지 않고 다시 학습하려면 다음 옵션을 추가한다.

```bash
python train.py --tapt on --new-words on --force-retrain-tapt
```

## 출력 파일

학습 결과는 기본적으로 `outputs_low_spec` 디렉터리에 저장된다.

```text
outputs_low_spec/
├── tapt_model/                 # TAPT 완료 모델
├── final_model/                # 최종 분류 모델 및 토크나이저
├── best_model_state.pt         # 검증 Macro F1 기준 최적 가중치
└── test_wrong_predictions.csv  # 테스트 세트 오분류 결과
```

`final_model/training_metadata.json`에는 학습 설정과 최종 테스트 지표가 기록된다.

## 실험 결과

독립 테스트 세트 1,086문장을 대상으로 평가한 결과는 다음과 같다.

| 모델            |   Accuracy |   Macro F1 |   Toxic F1 |   오분류 수 |
| ------------- | ---------: | ---------: | ---------: | ------: |
| 기본 모델         |     0.8103 |     0.8103 |     0.8075 |     206 |
| 신규 단어집 + TAPT | **0.8389** | **0.8387** | **0.8344** | **175** |

신규 단어집과 TAPT를 적용한 모델은 기본 모델보다 Macro F1-score가 약 **2.85%p** 증가하였으며, 전체 오분류 수는 206개에서 175개로 감소하였다.

혼동행렬 비교 결과에서도 False Positive는 143개에서 121개로, False Negative는 63개에서 54개로 감소하였다. 이를 통해 제안 방법이 정상 및 악성 클래스의 분류 성능을 모두 개선한 것을 확인하였다.

## 결론

실험 결과, 인터넷 채팅에서 빈번하게 사용되는 신조어, 축약어, 변형 표현 등을 반영한 신규 단어집과 채팅 말뭉치 기반 TAPT를 함께 적용했을 때 기본 모델보다 높은 분류 성능을 기록하였다.

따라서 한국어 실시간 채팅 환경인 경우, 일반 문서 중심으로 사전학습된 언어모델 보다 인터넷 채팅 특화 어휘와 비라벨 채팅 데이터를 이용한 TAPT 방식이 효과적인 방법이 될 수 있음을 확인하였다.
