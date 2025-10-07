from __future__ import annotations


def insert_entity_markers(sentence: str, subject: dict, obj: dict) -> str:
    """문장에 엔티티 마커를 삽입합니다. (단순 문자열 치환 기반)

    - 실제 KLUE 원본은 문자 오프셋이 있으므로 오프셋 기반 삽입이 바람직합니다.
    - 여기서는 word 필드가 문장 내에 그대로 등장한다고 가정합니다.
    """
    s_word = subject.get("word", "")
    s_type = subject.get("type", "ENT")
    o_word = obj.get("word", "")
    o_type = obj.get("type", "ENT")

    pairs = sorted(
        [(s_word, f"[SUBJ-{s_type}] {s_word} [/SUBJ-{s_type}]"), (o_word, f"[OBJ-{o_type}] {o_word} [/OBJ-{o_type}]")],
        key=lambda x: len(x[0]),
        reverse=True,
    )
    out = sentence
    for src, rep in pairs:
        if src:
            out = out.replace(src, rep, 1)
    return out


