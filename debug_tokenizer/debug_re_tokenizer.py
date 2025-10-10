"""
KLUE-RE 토크나이징 디버깅 스크립트

이 스크립트는 KLUE-RE 데이터셋의 토크나이징 과정을 자세히 확인할 수 있도록 합니다.
엔터티 마커 삽입과 토크나이징 과정을 시각적으로 출력합니다.
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

SPECIAL_TOKENS = [
    "<SUBJ:", ">", "</SUBJ>", "<OBJ:", "</OBJ>",
]

def insert_entity_markers(
    sentence: str,
    subj: dict,
    obj_: dict,
) -> str:
    """
    엔터티 마커를 삽입하는 함수 (원본과 동일)
    """
    s_start, s_end = int(subj["start_idx"]), int(subj["end_idx"])
    o_start, o_end = int(obj_["start_idx"]), int(obj_["end_idx"])

    first, second = ("subj", (s_start, s_end, subj)), ("obj", (o_start, o_end, obj_))
    if o_start < s_start:
        first, second = second, first

    def wrap(span_text: str, tag: str, ent_type: str) -> str:
        return f"<{tag}:{ent_type}>{span_text}</{tag}>"

    def apply(sentence_: str, start: int, end: int, tag: str, ent_type: str) -> tuple:
        pre = sentence_[:start]
        mid = sentence_[start : end + 1]
        post = sentence_[end + 1 :]
        wrapped = wrap(mid, tag, ent_type)
        new_sentence = pre + wrapped + post
        delta = len(wrapped) - len(mid)
        return new_sentence, delta

    sent = sentence
    if first[0] == "subj":
        sent, delta = apply(sent, first[1][0], first[1][1], "SUBJ", str(first[1][2]["type"]))
        s_shift = delta
        s2, e2 = second[1][0] + s_shift, second[1][1] + s_shift
        sent, _ = apply(sent, s2, e2, "OBJ", str(second[1][2]["type"]))
    else:
        sent, delta = apply(sent, first[1][0], first[1][1], "OBJ", str(first[1][2]["type"]))
        o_shift = delta
        s2, e2 = second[1][0] + o_shift, second[1][1] + o_shift
        sent, _ = apply(sent, s2, e2, "SUBJ", str(second[1][2]["type"]))

    return sent

def print_re_tokenization_details(tokenizer, samples, max_samples=3):
    """
    KLUE-RE 토크나이징 과정을 자세히 출력하는 함수
    
    Args:
        tokenizer: 사용할 토크나이저
        samples: KLUE-RE 샘플 데이터
        max_samples: 출력할 샘플 수
    """
    print("=" * 80)
    print("🔍 KLUE-RE 토크나이징 디버깅")
    print("=" * 80)
    
    for i in range(min(len(samples["sentence"]), max_samples)):
        print(f"\n📝 샘플 {i+1}")
        print("-" * 50)
        
        # 원본 데이터
        sentence = samples["sentence"][i]
        subj_entity = samples["subject_entity"][i]
        obj_entity = samples["object_entity"][i]
        label = samples["label"][i]
        
        print(f"원본 문장: {sentence}")
        print(f"주어 엔터티: {subj_entity}")
        print(f"목적어 엔터티: {obj_entity}")
        print(f"관계 라벨: {label}")
        
        # 엔터티 마커 삽입
        print(f"\n🏷️  엔터티 마커 삽입 과정:")
        print(f"원본: {sentence}")
        
        marked_sentence = insert_entity_markers(sentence, subj_entity, obj_entity)
        print(f"마커 삽입 후: {marked_sentence}")
        
        # 토크나이징 수행
        tokenized = tokenizer(
            marked_sentence,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        
        # 토큰 ID를 텍스트로 변환
        token_texts = tokenizer.convert_ids_to_tokens(input_ids)
        
        print(f"\n🔤 토크나이징 결과:")
        print(f"{'Index':<6} {'Token':<25} {'Token ID':<8} {'Attention':<10}")
        print("-" * 60)
        
        # 각 토큰 정보 출력
        for j, (token_text, token_id, attn) in enumerate(zip(token_texts, input_ids, attention_mask)):
            print(f"{j:<6} {token_text:<25} {token_id.item():<8} {attn.item():<10}")
        
        # 특수 토큰 확인
        print(f"\n🎯 특수 토큰 분석:")
        special_tokens = []
        for token_text in token_texts:
            if any(special in token_text for special in ["<SUBJ:", "<OBJ:", "</SUBJ>", "</OBJ>"]):
                special_tokens.append(token_text)
        
        if special_tokens:
            print(f"발견된 특수 토큰: {special_tokens}")
        else:
            print("특수 토큰이 발견되지 않았습니다.")
        
        # 최종 결과 요약
        print(f"\n📊 토크나이징 요약:")
        print(f"  • 원본 문장 길이: {len(sentence)} 문자")
        print(f"  • 마커 삽입 후 길이: {len(marked_sentence)} 문자")
        print(f"  • 토큰 수: {len(input_ids)}")
        print(f"  • 특수 토큰 수: {len(special_tokens)}")
        print(f"  • 패딩 토큰 수: {sum(1 for attn in attention_mask if attn == 0)}")

def main():
    """메인 함수"""
    print("🚀 KLUE-RE 토크나이징 디버깅 시작...")
    
    # 데이터셋 로드
    print("📥 데이터셋 로딩 중...")
    dataset = load_dataset("klue", "re")
    
    # 토크나이저 로드
    print("🔧 토크나이저 로딩 중...")
    model_name = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # 추가 마커 등록
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    
    # 관계 라벨 정보
    label_names = dataset["train"].features["label"].names
    print(f"📋 사용 가능한 관계 라벨: {label_names}")
    
    # 샘플 데이터 준비
    train_samples = dataset["train"].select(range(5))  # 처음 5개 샘플
    
    # 토크나이징 디버깅 실행
    print_re_tokenization_details(
        tokenizer=tokenizer,
        samples=train_samples,
        max_samples=3
    )
    
    print("\n✅ KLUE-RE 토크나이징 디버깅 완료!")

if __name__ == "__main__":
    main()
