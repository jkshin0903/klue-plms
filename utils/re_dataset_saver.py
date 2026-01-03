"""
KLUE-RE 토크나이징된 데이터셋 정보 저장 유틸리티

이 모듈은 KLUE-RE 태스크의 토크나이징된 데이터셋 정보를 파일로 저장하는 함수를 제공합니다.
"""

import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Any


def save_tokenized_dataset_info(tokenized_datasets, tokenizer, label_names, max_samples=10):
    """
    토크나이징된 데이터셋의 상세 정보를 파일로 저장하는 함수 (KLUE-RE용)
    
    Args:
        tokenized_datasets: 토크나이징된 데이터셋
        tokenizer: 사용된 토크나이저
        label_names: 관계 라벨 이름 리스트
        max_samples: 저장할 샘플 수
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"tokenized_dataset_info_re_{timestamp}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("🔍 KLUE-RE 토크나이징된 데이터셋 상세 정보\n")
        f.write("=" * 80 + "\n")
        f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"모델: klue/roberta-large\n")
        f.write(f"토크나이저: {tokenizer.__class__.__name__}\n\n")
        
        # 데이터셋 기본 정보
        f.write("📊 데이터셋 기본 정보:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Train 샘플 수: {len(tokenized_datasets['train'])}\n")
        if 'validation' in tokenized_datasets:
            f.write(f"Validation 샘플 수: {len(tokenized_datasets['validation'])}\n")
        if 'dev' in tokenized_datasets:
            f.write(f"Dev 샘플 수: {len(tokenized_datasets['dev'])}\n")
        
        f.write(f"관계 라벨 수: {len(label_names)}\n")
        f.write(f"관계 라벨 목록: {label_names}\n\n")
        
        # 데이터셋 구조 정보
        f.write("🏗️  데이터셋 구조:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Features: {list(tokenized_datasets['train'].features.keys())}\n")
        f.write(f"Column names: {tokenized_datasets['train'].column_names}\n\n")
        
        # 샘플별 상세 정보
        f.write("📝 샘플별 상세 정보:\n")
        f.write("=" * 80 + "\n")
        
        train_samples = tokenized_datasets["train"].select(range(max_samples))
        
        for i, sample in enumerate(train_samples):
            f.write(f"\n🔍 샘플 {i+1}:\n")
            f.write("-" * 50 + "\n")
            
            # 기본 정보
            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"]
            labels = sample["labels"]
            
            f.write(f"Input IDs 길이: {len(input_ids)}\n")
            f.write(f"Attention Mask 길이: {len(attention_mask)}\n")
            f.write(f"Labels: {labels}\n")
            
            # 토큰별 상세 정보
            token_texts = tokenizer.convert_ids_to_tokens(input_ids)
            
            f.write(f"\n토큰별 상세 정보:\n")
            f.write(f"{'Index':<6} {'Token':<25} {'Token ID':<10} {'Attention':<10}\n")
            f.write("-" * 60 + "\n")
            
            for j, (token_text, token_id, attn) in enumerate(
                zip(token_texts, input_ids, attention_mask)
            ):
                f.write(f"{j:<6} {token_text:<25} {token_id:<10} {attn:<10}\n")
            
            # 특수 토큰 확인
            special_tokens = []
            for token_text in token_texts:
                if any(special in token_text for special in ["<SUBJ:", "<OBJ:", "</SUBJ>", "</OBJ>"]):
                    special_tokens.append(token_text)
            
            # 통계 정보
            f.write(f"\n📊 샘플 통계:\n")
            f.write(f"  • 총 토큰 수: {len(input_ids)}\n")
            f.write(f"  • 특수 토큰 수: {len(special_tokens)}\n")
            f.write(f"  • 패딩 토큰 수: {sum(1 for attn in attention_mask if attn == 0)}\n")
            
            # 원본 텍스트 복원
            original_text = tokenizer.decode(input_ids, skip_special_tokens=True)
            f.write(f"  • 복원된 텍스트: {original_text}\n")
            
            if special_tokens:
                f.write(f"  • 발견된 특수 토큰: {special_tokens}\n")
        
        # 전체 데이터셋 통계
        f.write(f"\n\n📈 전체 데이터셋 통계:\n")
        f.write("=" * 80 + "\n")
        
        # 모든 샘플의 길이 통계
        all_lengths = [len(sample["input_ids"]) for sample in tokenized_datasets["train"]]
        f.write(f"토큰 길이 통계:\n")
        f.write(f"  • 평균 길이: {np.mean(all_lengths):.2f}\n")
        f.write(f"  • 최소 길이: {min(all_lengths)}\n")
        f.write(f"  • 최대 길이: {max(all_lengths)}\n")
        f.write(f"  • 중간값: {np.median(all_lengths):.2f}\n")
        
        # 라벨 분포
        f.write(f"\n라벨 분포:\n")
        label_counts = {}
        for sample in tokenized_datasets["train"]:
            label = sample["labels"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        for label_id, count in sorted(label_counts.items()):
            label_name = label_names[label_id]
            f.write(f"  • {label_name} (ID: {label_id}): {count}개\n")
    
    print(f"✅ 토크나이징된 데이터셋 정보가 '{output_file}' 파일에 저장되었습니다.")
    print(f"📁 파일 위치: {os.path.abspath(output_file)}")
