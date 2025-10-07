from __future__ import annotations

import os
from typing import Dict, List, Optional


def read_tsv_generic(path: str) -> List[Dict[str, str]]:
    """TSV 파일을 읽어 Dict 리스트로 반환합니다. (헤더 필요)

    각 행은 문자열 값의 딕셔너리입니다.
    파일이 없으면 빈 리스트를 반환합니다.
    """
    import csv
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append(dict(r))
    return rows


def parse_space_separated_list(value: Optional[str]) -> List[str]:
    """공백으로 구분된 문자열을 토큰 리스트로 변환합니다. None/빈 문자열은 빈 리스트."""
    if not value:
        return []
    return value.strip().split()


def parse_space_separated_int_list(value: Optional[str]) -> List[int]:
    """공백 구분 정수 리스트 파서. 비어있으면 빈 리스트."""
    items = parse_space_separated_list(value)
    return [int(x) for x in items] if items else []


def read_dp_tsv(path: str) -> List[Dict]:
    """DP용 TSV 로더.

    지원 포맷 2가지:
    1) 공식 KLUE-DP v1.1 포맷 (문장 시작은 '## klue-dp-...' 주석 라인, 그 후 토큰 단위 행)
       칼럼: INDEX WORD_FORM LEMMA POS HEAD DEPREL
    2) 간단 헤더형(한 행 = 한 문장) 포맷: columns: tokens, heads, deprels (모두 공백 구분)

    - heads는 1-based, 0은 ROOT
    """
    if not os.path.exists(path):
        return []

    # 먼저 파일 헤더를 검사하여 공식 포맷 여부를 판별
    with open(path, "r", encoding="utf-8") as f:
        head_lines = []
        for _ in range(50):
            line = f.readline()
            if not line:
                break
            head_lines.append(line.rstrip("\n"))

    def looks_like_official(lines: List[str]) -> bool:
        for ln in lines:
            if ln.startswith("## klue-dp-") or "## 칼럼명" in ln:
                return True
        return False

    if looks_like_official(head_lines):
        # 공식 포맷 파싱: 문장 경계는 '## '로 시작하는 라인들
        sentences: List[Dict] = []
        tokens: List[str] = []
        heads: List[int] = []
        deprels: List[str] = []

        def flush():
            if tokens:
                sentences.append({"tokens": tokens.copy(), "heads": heads.copy(), "deprels": deprels.copy()})
            tokens.clear()
            heads.clear()
            deprels.clear()

        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("## "):
                    # 새 문장 시작 신호. 이전 문장을 flush
                    flush()
                    continue
                # 토큰 행: INDEX WORD_FORM LEMMA POS HEAD DEPREL
                parts = line.split()
                if len(parts) < 6:
                    # 예외적 포맷은 스킵
                    continue
                # parts[0]=index, parts[1]=WORD_FORM, parts[4]=HEAD, parts[5]=DEPREL
                tokens.append(parts[1])
                try:
                    heads.append(int(parts[4]))
                except Exception:
                    heads.append(0)
                deprels.append(parts[5])
        # 마지막 문장 flush
        flush()
        return sentences

    # 아니면 간단 헤더형 포맷 처리
    rows = read_tsv_generic(path)
    out: List[Dict] = []
    for r in rows:
        tokens = parse_space_separated_list(r.get("tokens"))
        heads = parse_space_separated_int_list(r.get("heads"))
        deprels = parse_space_separated_list(r.get("deprels"))
        out.append({"tokens": tokens, "heads": heads, "deprels": deprels})
    return out


def read_ner_tsv(path: str) -> List[Dict]:
    """NER용 TSV 로더. columns: tokens, tags (둘 다 공백 구분)."""
    rows = read_tsv_generic(path)
    out: List[Dict] = []
    for r in rows:
        tokens = parse_space_separated_list(r.get("tokens"))
        tags = parse_space_separated_list(r.get("tags"))
        out.append({"tokens": tokens, "tags": tags})
    return out


