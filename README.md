# KLUE PLMs 파인튜닝 (NER / RE / DP)

이 저장소는 KLUE 공개 PLM(`klue/roberta-large`)을 사용하여 아래 태스크들을 파인튜닝하기 위한 간단한 스크립트를 제공합니다.

- NER: `ner.py`
- RE(관계 추출): `re.py`
- DP(의존 구문 분석): `dp.py`

KLUE 데이터 포맷과 대체로 호환되며, 필요한 최소 전처리/정렬 로직을 포함합니다.

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CUDA 환경이면 PyTorch를 환경에 맞는 버전으로 별도 설치하세요.

## 환경 변수

- `MODEL_NAME`: 기본값은 `klue/roberta-large`. 다른 체크포인트 사용 시 설정합니다.

## 데이터 포맷

- NER (`data/klue-ner` 예시)
  - `train.json`, `dev.json`, `test.json`
  - 각 샘플은 `{ "tokens": [..], "tags": [..] }` 또는 원본 KLUE 엔티티 포맷을 지원합니다.

- RE (`data/klue-re` 예시)
  - 각 샘플: `{ sentence, subject_entity, object_entity, label }`
  - 문장 내 엔티티는 스크립트에서 특수 마커로 감싸 토크나이즈합니다.

- DP (`data/klue-dp` 예시)
  - 각 샘플: `{ tokens, heads, deprels }`
  - `heads` 는 단어 단위 1-based 인덱스, 0은 ROOT를 의미합니다.

## 실행 예시

- NER
```bash
python ner.py \
  --data_dir ./data/klue-ner \
  --output_dir ./outputs/ner \
  --num_train_epochs 5 \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --cuda_visible_devices 0 \
  --device_index 0
```

- RE
```bash
python re.py \
  --data_dir ./data/klue-re \
  --output_dir ./outputs/re \
  --num_train_epochs 5 \
  --per_device_train_batch_size 32 \
  --learning_rate 3e-5 \
  --cuda_visible_devices 0 \
  --device_index 0
```

- DP
```bash
python dp.py \
  --data_dir ./data/klue-dp \
  --output_dir ./outputs/dp \
  --num_train_epochs 20 \
  --per_device_train_batch_size 16 \
  --learning_rate 3e-5 \
  --cuda_visible_devices 0 \
  --device_index 0
```

### CUDA 디버깅 옵션

- 기본적으로 `CUDA_LAUNCH_BLOCKING=1`, `TORCH_USE_CUDA_DSA=1` 가 활성화됩니다.
- 비활성화하려면 `--no_launch_blocking`, `--no_cuda_dsa` 플래그를 사용하세요.

## 결과 저장

- 각 태스크의 출력 디렉토리에 라벨 목록(`labels.json` 또는 `deprels.json`)과 모델 가중치가 저장됩니다.

## 주의사항

- 본 스크립트는 학습 파이프라인을 빠르게 구축하기 위한 최소 구현입니다. 실제 대회/프로덕션 목적이면
  - 문자 오프셋 기반의 정밀 토큰-엔티티 정렬
  - 데이터 검증/에러 처리 강화
  - UAS/LAS 등 보다 정교한 평가지표 및 리포팅
  등을 보강하시길 권장합니다.
