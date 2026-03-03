# Korean LM Attention Analysis for NER (KLUE RoBERTa)

본 레포지토리는 한국어의 교착어적 특성과 형태론적 복잡성이 언어모델 내부 표현(특히 Transformer attention)에 어떻게 반영되는지 분석한 연구(한국컴퓨터정보학회논문지, 2026)의 재현 및 확장 실험을 위한 코드/스크립트를 제공합니다.

- Fine-tuning: KLUE RoBERTa Base + NER
- Analysis: attention weight 추출, 토큰 간 attention 강도 시각화, 패턴별 attention 분포 정량화
- Key patterns (논문 요약 기반):
  - Span-internal cohesion (엔티티 내부 토큰의 경계 집중)
  - Boundary alignment (엔티티 직후 조사/비엔티티 토큰이 경계 신호)
  - Long-distance dependencies (원거리 논항 간 의미적 결속)

## Paper

- DBpia: https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12569204  
- DOI: 10.9708/jksci.2026.31.01.041
- Venue: 한국컴퓨터정보학회논문지, 31(1), 41–49 (2026)

> 이 레포지토리는 연구 재현을 목적으로 하며, 논문/데이터/모델의 라이선스를 준수해야 합니다.

## What’s Inside

- NER 학습 파이프라인
  - 문자 단위 라벨을 고려한 서브워드-라벨 정렬(subword–label alignment)
  - 마스킹 기반의 안정적 학습 설계(라벨 누락/불일치 완화 목적)
- Attention 추출
  - layer/head 별 attention weight 저장
  - 문장/스팬 단위의 attention 서브셋 추출
- 시각화 & 정량 분석
  - 토큰 간 attention heatmap
  - 패턴(응집/경계/장거리)별 attention 분포 통계 산출

## Repository Structure (suggested)

아래는 기본 디렉토리 구조 예시입니다. 실제 구현에 맞게 수정하세요.

- `configs/` : 학습/분석 설정(yaml/json)
- `data/`
  - `raw/` : 원본 데이터(미포함 권장)
  - `processed/` : 전처리/정렬 결과
- `src/`
  - `train_ner.py` : NER fine-tuning 엔트리포인트
  - `align_labels.py` : subword–label alignment 유틸
  - `extract_attention.py` : attention weight 추출
  - `analyze_patterns.py` : 패턴별 정량 분석
  - `viz_attention.py` : heatmap 등 시각화
- `notebooks/` : 분석/그림 재현 노트북
- `outputs/`
  - `checkpoints/`
  - `attentions/`
  - `figures/`

## Environment

권장: Python 3.10+

예시(필요에 맞게 조정):
- PyTorch
- Hugging Face `transformers`, `datasets`
- `numpy`, `pandas`
- `matplotlib`/`seaborn` 또는 `plotly`

### Setup

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows) .venv\Scripts\activate
pip install -r requirements.txt
