"""
KLUE-DP 토크나이징 디버깅 스크립트

이 스크립트는 KLUE-DP 데이터셋의 토크나이징 과정을 자세히 확인할 수 있도록 합니다.
단어-서브토큰 정렬과 의존구문 분석 라벨 매핑 과정을 시각적으로 출력합니다.
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

def build_labels_and_align_debug(
    tokenizer,
    tokens: list,
    heads: list,
    deprels: list,
) -> dict:
    """
    의존구문 분석 라벨 정렬 과정을 디버깅용으로 구현
    """
    print(f"  📝 단어 토큰: {tokens}")
    print(f"  🎯 Head 인덱스: {heads}")
    print(f"  🏷️  Deprel 라벨: {deprels}")
    
    enc = tokenizer(tokens, is_split_into_words=True, truncation=True)
    word_ids = enc.word_ids()  # 개별 토큰이 속한 단어의 인덱스
    
    print(f"  🔗 Word IDs: {word_ids}")
    
    # 단어 인덱스 -> 첫 서브토큰 인덱스 매핑
    word_to_first_subtok = {}
    for i, w in enumerate(word_ids):
        if w is None: 
            continue  # 특수 토큰(CLS/SEP), 패딩 토큰 무시
        if w not in word_to_first_subtok:
            word_to_first_subtok[w] = i
    
    print(f"  📍 단어-서브토큰 매핑: {word_to_first_subtok}")
    
    seq_len = len(enc["input_ids"])
    labels_head = [IGNORE_INDEX] * seq_len
    labels_deprel = [IGNORE_INDEX] * seq_len
    
    # 각 단어의 첫 서브토큰에만 라벨을 부여
    for w_idx in range(len(tokens)):
        if w_idx not in word_to_first_subtok: 
            continue
        tok_i = word_to_first_subtok[w_idx]
        gold_head_word = int(heads[w_idx])
        # 정답 head가 ROOT를 가리키면 CLS 토큰(0)으로 매핑
        gold_head_tok = 0 if gold_head_word <= 0 else word_to_first_subtok.get(gold_head_word, 0)
        labels_head[tok_i] = gold_head_tok
        labels_deprel[tok_i] = int(deprels[w_idx])
    
    enc["labels_head"] = labels_head
    enc["labels_deprel"] = labels_deprel
    return enc

def print_dp_tokenization_details(tokenizer, samples, max_samples=3):
    """
    KLUE-DP 토크나이징 과정을 자세히 출력하는 함수
    
    Args:
        tokenizer: 사용할 토크나이저
        samples: KLUE-DP 샘플 데이터
        max_samples: 출력할 샘플 수
    """
    print("=" * 80)
    print("🔍 KLUE-DP 토크나이징 디버깅")
    print("=" * 80)
    
    for i in range(min(len(samples["word_form"]), max_samples)):
        print(f"\n📝 샘플 {i+1}")
        print("-" * 50)
        
        # 원본 데이터
        tokens = samples["word_form"][i]
        heads = samples["head"][i]
        deprels = samples["deprel"][i]
        
        print(f"원본 단어: {tokens}")
        print(f"Head 인덱스: {heads}")
        print(f"Deprel 라벨: {deprels}")
        
        # 토크나이징 및 라벨 정렬
        print(f"\n🔄 토크나이징 및 라벨 정렬 과정:")
        enc = build_labels_and_align_debug(tokenizer, tokens, heads, deprels)
        
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels_head = enc["labels_head"]
        labels_deprel = enc["labels_deprel"]
        
        # 토큰 ID를 텍스트로 변환
        token_texts = tokenizer.convert_ids_to_tokens(input_ids)
        
        print(f"\n🔤 서브토큰 분석:")
        print(f"{'Index':<6} {'Token':<20} {'Token ID':<8} {'Head Label':<12} {'Deprel Label':<12}")
        print("-" * 70)
        
        # 각 서브토큰 정보 출력
        for j, (token_text, token_id, head_label, deprel_label) in enumerate(
            zip(token_texts, input_ids, labels_head, labels_deprel)
        ):
            head_str = str(head_label) if head_label != IGNORE_INDEX else "IGNORE"
            deprel_str = str(deprel_label) if deprel_label != IGNORE_INDEX else "IGNORE"
            print(f"{j:<6} {token_text:<20} {token_id:<8} {head_str:<12} {deprel_str:<12}")
        
        # 의존관계 시각화
        print(f"\n🌳 의존관계 구조:")
        print(f"{'Word':<15} {'Head':<8} {'Deprel':<12} {'Subtoken':<20}")
        print("-" * 60)
        
        word_to_first_subtok = {}
        word_ids = tokenizer(tokens, is_split_into_words=True, truncation=True).word_ids()
        for j, w in enumerate(word_ids):
            if w is None: continue
            if w not in word_to_first_subtok:
                word_to_first_subtok[w] = j
        
        for w_idx in range(len(tokens)):
            if w_idx in word_to_first_subtok:
                tok_idx = word_to_first_subtok[w_idx]
                word_text = tokens[w_idx]
                head_word = heads[w_idx]
                deprel_label = deprels[w_idx]
                subtoken_text = token_texts[tok_idx]
                print(f"{word_text:<15} {head_word:<8} {deprel_label:<12} {subtoken_text:<20}")
        
        # 최종 결과 요약
        print(f"\n📊 토크나이징 요약:")
        print(f"  • 원본 단어 수: {len(tokens)}")
        print(f"  • 서브토큰 수: {len(input_ids)}")
        print(f"  • 라벨이 할당된 서브토큰 수: {sum(1 for label in labels_head if label != IGNORE_INDEX)}")
        print(f"  • 특수 토큰 수: {sum(1 for w in word_ids if w is None)}")

def main():
    """메인 함수"""
    print("🚀 KLUE-DP 토크나이징 디버깅 시작...")
    
    # 데이터셋 로드
    print("📥 데이터셋 로딩 중...")
    dataset = load_dataset("klue", "dp")
    
    # 토크나이저 로드
    print("🔧 토크나이저 로딩 중...")
    model_name = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # 의존관계 라벨 정보
    rel_feature = dataset["train"].features["deprel"].feature
    num_relations = len(rel_feature.names)
    print(f"📋 사용 가능한 의존관계 라벨 수: {num_relations}")
    print(f"📋 의존관계 라벨: {rel_feature.names}")
    
    # 샘플 데이터 준비
    train_samples = dataset["train"].select(range(5))  # 처음 5개 샘플
    
    # 토크나이징 디버깅 실행
    print_dp_tokenization_details(
        tokenizer=tokenizer,
        samples=train_samples,
        max_samples=3
    )
    
    print("\n✅ KLUE-DP 토크나이징 디버깅 완료!")

if __name__ == "__main__":
    main()
