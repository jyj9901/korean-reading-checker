import os
import re
from difflib import SequenceMatcher

from flask import Flask, jsonify, request, send_from_directory
from openai import OpenAI

app = Flask(__name__, static_folder="public")
client = OpenAI()  # reads OPENAI_API_KEY from the environment
STT_MODEL = os.getenv("STT_MODEL", "whisper-1")

# 발음형 변환(선택): pip install g2pk 가 되어 있으면 자동 사용
try:
    from g2pk import G2p
    _g2p = G2p()
except Exception:
    _g2p = None

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
KEEP = r"[0-9A-Za-z가-힣]"


def normalize(text):
    """띄어쓰기·문장부호 제거"""
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text)


def to_jamo(text):
    out = []
    for ch in text:
        c = ord(ch) - 0xAC00
        if 0 <= c < 11172:
            out += [CHO[c // 588], JUNG[(c % 588) // 28]]
            if JONG[c % 28] != " ":
                out.append(JONG[c % 28])
        else:
            out.append(ch)
    return out


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def accuracy(ref, hyp):
    """자모 단위 편집 거리 기반 정확도(0~100). 발음형이 있으면 더 유리한 쪽으로 채점."""
    candidates = [ref]
    if _g2p:
        candidates.append(_g2p(ref))
    h = to_jamo(normalize(hyp))
    best = 0
    for c in candidates:
        r = to_jamo(normalize(c))
        if r:
            best = max(best, round((1 - lev(r, h) / len(r)) * 100))
    return max(0, min(100, best))


def mark(ref, hyp):
    """기준 문장의 각 글자가 인식 결과와 일치하는지 표시"""
    r, h = normalize(ref), normalize(hyp)
    ok = [False] * len(r)
    for blk in SequenceMatcher(None, r, h, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            ok[blk.a + k] = True
    out, i = [], 0
    for c in ref:
        if re.match(KEEP, c):
            out.append({"ch": c, "ok": ok[i]})
            i += 1
        else:
            out.append({"ch": c, "ok": True})
    return out


@app.get("/")
def index():
    return send_from_directory("public", "index.html")


@app.post("/api/check")
def check():
    ref = request.form.get("reference", "").strip()
    f = request.files.get("audio")
    if not ref or not f:
        return jsonify(error="기준 문장과 녹음 파일이 필요합니다."), 400
    mt = f.mimetype or ""
    ext = "mp4" if "mp4" in mt else "ogg" if "ogg" in mt else "webm"
    try:
        tr = client.audio.transcriptions.create(
            model=STT_MODEL, file=(f"rec.{ext}", f.read()), language="ko"
        )
    except Exception as e:
        return jsonify(error=f"음성 인식 실패: {e}"), 500
    hyp = tr.text
    return jsonify(transcript=hyp, score=accuracy(ref, hyp), marks=mark(ref, hyp))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
