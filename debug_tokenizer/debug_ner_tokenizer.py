"""
KLUE-NER 토크나이징 디버깅 스크립트

이 스크립트는 KLUE-NER 데이터셋의 토크나이징 과정을 자세히 확인할 수 있도록 합니다.
개별 토큰의 정보(토큰 텍스트, 토큰 ID, 라벨, 오프셋 매핑 등)를 시각적으로 출력합니다.
"""

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

device = torch.device(f'cuda:0')
torch.cuda.set_device(device)

# 라벨 마스킹 상수
IGNORE_INDEX: int = -100

def print_ner_tokenization_details(tokenizer, sample_text, tokens, ner_tags, max_samples=3):
    """
    KLUE-NER 토크나이징 과정을 자세히 출력하는 함수
    
    Args:
        tokenizer: 사용할 토크나이저
        sample_text: 원본 텍스트
        tokens: 단어 단위 토큰 리스트
        ner_tags: NER 태그 리스트
        max_samples: 출력할 샘플 수
    """
    print("=" * 80)
    print("🔍 KLUE-NER 토크나이징 디버깅")
    print("=" * 80)
    
    for i in range(min(len(sample_text), max_samples)):
        print(f"\n📝 샘플 {i+1}")
        print("-" * 50)
        
        # 원본 데이터
        text = sample_text[i]
        word_tokens = tokens[i]
        word_labels = ner_tags[i]
        
        print(f"원본 텍스트: {text}")
        print(f"단어 토큰: {word_tokens}")
        print(f"단어 라벨: {word_labels}")
        
        # 토크나이징 수행
        tokenized = tokenizer(
            text,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        offset_mapping = tokenized["offset_mapping"][0]
        
        # 토큰 ID를 텍스트로 변환
        token_texts = tokenizer.convert_ids_to_tokens(input_ids)
        
        print(f"\n🔤 서브토큰 분석:")
        print(f"{'Index':<6} {'Token':<20} {'Token ID':<8} {'Offset':<15} {'Type':<10}")
        print("-" * 70)
        
        # 각 서브토큰 정보 출력
        for j, (token_text, token_id, offset) in enumerate(zip(token_texts, input_ids, offset_mapping)):
            start, end = offset
            token_type = "SPECIAL" if start == end else "WORD"
            
            print(f"{j:<6} {token_text:<20} {token_id.item():<8} {f'({start},{end})':<15} {token_type:<10}")
        
        # 라벨 정렬 과정 시뮬레이션
        print(f"\n🏷️  라벨 정렬 과정:")
        print(f"{'Word':<15} {'Word Label':<12} {'Subtoken':<20} {'Subtoken Label':<15}")
        print("-" * 70)
        
        # 단어-서브토큰 매핑 생성
        word_to_first_subtok = {}
        for j, offset in enumerate(offset_mapping):
            start, end = offset
            if start == end:  # 특수 토큰
                continue
            
            # 이 서브토큰이 커버하는 첫 번째 단어 찾기
            char_pos = start
            word_idx = None
            current_pos = 0
            
            for k, word in enumerate(word_tokens):
                if current_pos <= char_pos < current_pos + len(word):
                    word_idx = k
                    break
                current_pos += len(word) + 1  # +1 for space
            
            if word_idx is not None and word_idx not in word_to_first_subtok:
                word_to_first_subtok[word_idx] = j
        
        # 라벨 할당 시뮬레이션
        subtoken_labels = [IGNORE_INDEX] * len(input_ids)
        for word_idx, subtoken_idx in word_to_first_subtok.items():
            if word_idx < len(word_labels):
                subtoken_labels[subtoken_idx] = word_labels[word_idx]
                word_text = word_tokens[word_idx] if word_idx < len(word_tokens) else "UNK"
                subtoken_text = token_texts[subtoken_idx]
                print(f"{word_text:<15} {word_labels[word_idx]:<12} {subtoken_text:<20} {subtoken_labels[subtoken_idx]:<15}")
        
        # 최종 결과 요약
        print(f"\n📊 토크나이징 요약:")
        print(f"  • 원본 단어 수: {len(word_tokens)}")
        print(f"  • 서브토큰 수: {len(input_ids)}")
        print(f"  • 라벨이 할당된 서브토큰 수: {sum(1 for label in subtoken_labels if label != IGNORE_INDEX)}")
        print(f"  • 특수 토큰 수: {sum(1 for offset in offset_mapping if offset[0] == offset[1])}")

def main():
    """메인 함수"""
    print("🚀 KLUE-NER 토크나이징 디버깅 시작...")
    
    # 데이터셋 로드
    print("📥 데이터셋 로딩 중...")
    dataset = load_dataset("klue", "ner")
    
    # 토크나이저 로드
    print("🔧 토크나이저 로딩 중...")
    model_name = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # NER 라벨 정보
    label_names = dataset["train"].features["ner_tags"].feature.names
    print(f"📋 사용 가능한 NER 라벨: {label_names}")
    
    # 샘플 데이터 준비
    train_samples = dataset["train"].select(range(5))  # 처음 5개 샘플
    
    # 원본 텍스트로 변환 (단어 리스트를 문장으로 결합)
    sample_texts = [" ".join(tokens) for tokens in train_samples["tokens"]]
    
    # 토크나이징 디버깅 실행
    print_ner_tokenization_details(
        tokenizer=tokenizer,
        sample_text=sample_texts,
        tokens=train_samples["tokens"],
        ner_tags=train_samples["ner_tags"],
        max_samples=3
    )
    
    print("\n✅ KLUE-NER 토크나이징 디버깅 완료!")

if __name__ == "__main__":
    main()
