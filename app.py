import os
import re
import secrets
import uuid
from difflib import SequenceMatcher

import requests
from flask import Flask, Response, jsonify, request, send_from_directory
from openai import OpenAI

app = Flask(__name__, static_folder="public")
# 기본은 OpenAI. Groq 등 OpenAI 호환 서비스를 쓰려면 환경 변수만 바꾸면 됩니다.
#   STT_API_KEY  : 서비스 키 (없으면 OPENAI_API_KEY 사용)
#   STT_BASE_URL : 예) https://api.groq.com/openai/v1
#   STT_MODEL    : 예) whisper-large-v3
client = OpenAI(
    api_key=os.getenv("STT_API_KEY") or os.getenv("OPENAI_API_KEY") or "missing",
    base_url=os.getenv("STT_BASE_URL") or None,
)
STT_MODEL = os.getenv("STT_MODEL", "whisper-1")

SB_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SB_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
TEACHER_PW = os.getenv("TEACHER_PASSWORD", "")

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


def sb(method, path, **kw):
    h = {"apikey": SB_KEY}
    if not SB_KEY.startswith("sb_"):  # 예전 service_role(JWT) 키만 Authorization 헤더가 필요
        h["Authorization"] = f"Bearer {SB_KEY}"
    h.update(kw.pop("headers", {}))
    r = requests.request(method, f"{SB_URL}{path}", headers=h, timeout=20, **kw)
    r.raise_for_status()
    return r


def is_teacher():
    pw = request.headers.get("X-Teacher-Password", "")
    return bool(TEACHER_PW) and secrets.compare_digest(pw.encode(), TEACHER_PW.encode())


def deny():
    return jsonify(error="선생님 비밀번호가 올바르지 않습니다."), 401


def find_assignment(code):
    code = (code or "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{6}", code):
        return None
    rows = sb("GET", "/rest/v1/assignments",
              params={"code": f"eq.{code}", "select": "id,code,title,sentence"}).json()
    return rows[0] if rows else None


@app.errorhandler(requests.RequestException)
def db_error(e):
    return jsonify(error="데이터베이스 오류가 발생했습니다."), 500


@app.get("/teacher")
def teacher_page():
    return send_from_directory("public", "teacher.html")


@app.get("/api/assignment/<code>")
def get_assignment(code):
    a = find_assignment(code)
    if not a:
        return jsonify(error="숙제 코드를 찾을 수 없습니다."), 404
    return jsonify(title=a["title"], sentence=a["sentence"])


@app.post("/api/submit")
def submit():
    a = find_assignment(request.form.get("code"))
    student = request.form.get("student", "").strip()[:50]
    f = request.files.get("audio")
    if not a:
        return jsonify(error="숙제 코드를 찾을 수 없습니다."), 404
    if not student or not f:
        return jsonify(error="이름과 녹음이 필요합니다."), 400
    data, mt = f.read(), f.mimetype or ""
    ext = "mp4" if "mp4" in mt else "ogg" if "ogg" in mt else "webm"
    try:
        hyp = client.audio.transcriptions.create(
            model=STT_MODEL, file=(f"rec.{ext}", data), language="ko"
        ).text
    except Exception as e:
        return jsonify(error=f"음성 인식 실패: {e}"), 500
    score = accuracy(a["sentence"], hyp)
    path = f"{a['id']}/{uuid.uuid4().hex}.{ext}"
    try:
        sb("POST", f"/storage/v1/object/recordings/{path}", data=data,
           headers={"Content-Type": mt or "application/octet-stream"})
    except requests.RequestException:
        path = None  # 녹음 저장이 실패해도 점수는 기록
    sb("POST", "/rest/v1/submissions", json={
        "assignment_id": a["id"], "student": student,
        "score": score, "transcript": hyp, "audio_path": path})
    return jsonify(transcript=hyp, score=score, marks=mark(a["sentence"], hyp))


@app.post("/api/assignments")
def create_assignment():
    if not is_teacher():
        return deny()
    d = request.get_json(silent=True) or {}
    title, sentence = d.get("title", "").strip(), d.get("sentence", "").strip()
    if not title or not sentence:
        return jsonify(error="제목과 문장을 입력하세요."), 400
    row = sb("POST", "/rest/v1/assignments",
             json={"code": secrets.token_hex(3).upper(), "title": title, "sentence": sentence},
             headers={"Prefer": "return=representation"}).json()[0]
    return jsonify(row)


@app.get("/api/assignments")
def list_assignments():
    if not is_teacher():
        return deny()
    return jsonify(sb("GET", "/rest/v1/assignments",
                      params={"select": "*", "order": "created_at.desc"}).json())


@app.get("/api/submissions")
def list_submissions():
    if not is_teacher():
        return deny()
    params = {"select": "id,student,score,transcript,audio_path,created_at",
              "order": "created_at.desc"}
    aid = request.args.get("assignment_id", "")
    if aid.isdigit():
        params["assignment_id"] = f"eq.{aid}"
    return jsonify(sb("GET", "/rest/v1/submissions", params=params).json())


@app.get("/api/audio/<int:sid>")
def get_audio(sid):
    if not is_teacher():
        return deny()
    rows = sb("GET", "/rest/v1/submissions",
              params={"id": f"eq.{sid}", "select": "audio_path"}).json()
    if not rows or not rows[0]["audio_path"]:
        return jsonify(error="저장된 녹음이 없습니다."), 404
    p = rows[0]["audio_path"]
    r = sb("GET", f"/storage/v1/object/recordings/{p}")
    mime = {"mp4": "audio/mp4", "ogg": "audio/ogg"}.get(p.rsplit(".", 1)[-1], "audio/webm")
    return Response(r.content, mimetype=mime)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
