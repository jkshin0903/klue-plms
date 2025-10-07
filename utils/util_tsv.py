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
    """DP용 TSV 로더. columns: tokens, heads, deprels (모두 공백 구분).

    - heads는 1-based, 0은 ROOT
    """
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


