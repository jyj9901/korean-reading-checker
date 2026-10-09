import os
import re
import secrets
import uuid
import calendar
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

import requests
from flask import Flask, Response, jsonify, request, send_from_directory
from openai import OpenAI
from werkzeug.exceptions import HTTPException

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
TEACHER_PW = os.getenv("TEACHER_PASSWORD", "")  # 관리자 비밀번호(선생님 사용 기간 관리용)
SB_ANON = os.getenv("SUPABASE_ANON_KEY", "")

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


def me():
    """로그인한 선생님 (user, profile). 토큰이 없거나 틀리면 (None, None)"""
    tok = request.headers.get("Authorization", "")
    tok = tok[7:].strip() if tok.startswith("Bearer ") else ""
    if not tok:
        return None, None
    r = requests.get(f"{SB_URL}/auth/v1/user", timeout=10,
                     headers={"apikey": SB_ANON or SB_KEY, "Authorization": f"Bearer {tok}"})
    if r.status_code != 200:
        return None, None
    u = r.json()
    rows = sb("GET", "/rest/v1/teachers", params={"id": f"eq.{u['id']}", "select": "*"}).json()
    return u, (rows[0] if rows else None)


def active(p):
    try:
        return datetime.fromisoformat(p["paid_until"].replace("Z", "+00:00")) > datetime.now(timezone.utc)
    except Exception:
        return False


def my_class_ids(u):
    rows = sb("GET", "/rest/v1/classes", params={"teacher_id": f"eq.{u['id']}", "select": "id"}).json()
    return [r["id"] for r in rows]


def owns(u, aid):
    ids = my_class_ids(u)
    return bool(ids) and bool(sb("GET", "/rest/v1/assignments", params={
        "id": f"eq.{aid}", "class_id": "in.(" + ",".join(map(str, ids)) + ")", "select": "id"}).json())


def drop_assignments(ids):
    """과제와 그 제출 기록·녹음 파일을 모두 삭제"""
    if not ids:
        return
    inl = "in.(" + ",".join(str(int(i)) for i in ids) + ")"
    subs = sb("GET", "/rest/v1/submissions", params={"assignment_id": inl, "select": "audio_path"}).json()
    paths = [s["audio_path"] for s in subs if s.get("audio_path")]
    for i in range(0, len(paths), 100):
        try:
            sb("DELETE", "/storage/v1/object/recordings", json={"prefixes": paths[i:i + 100]})
        except requests.RequestException:
            pass  # 파일 삭제가 실패해도 기록 삭제는 진행
    sb("DELETE", "/rest/v1/submissions", params={"assignment_id": inl})
    sb("DELETE", "/rest/v1/assignments", params={"id": inl})


def admin_user(u):
    """ADMIN_EMAILS(쉼표로 구분)에 있고, 이메일 인증이 끝난 계정만 관리자"""
    emails = {e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()}
    return bool(u and u.get("email_confirmed_at") and (u.get("email") or "").lower() in emails)


def is_admin():
    return admin_user(me()[0])


def deny():
    return jsonify(error="로그인이 필요합니다."), 401


def not_paid():
    return jsonify(error="사용 기간이 아닙니다. 관리자에게 문의하세요."), 402


CODE_CHARS = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # 헷갈리는 글자(0/O, 1/I/L) 제외


def new_code():
    return "".join(secrets.choice(CODE_CHARS) for _ in range(6))


def assign_code(cid):
    """반에 새 입장 코드를 부여(겹치면 다시 시도)"""
    for _ in range(5):
        code = new_code()
        try:
            sb("PATCH", f"/rest/v1/classes?id=eq.{cid}", json={"join_code": code})
            return code
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 409:
                raise
    raise RuntimeError("반 코드를 만들지 못했습니다.")


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


@app.errorhandler(Exception)
def any_error(e):  # 예상 못 한 오류도 화면에 이유가 보이도록 JSON으로 돌려줌
    if isinstance(e, HTTPException):
        return (jsonify(error=f"{e.code} {e.name}"), e.code) if request.path.startswith("/api/") else e
    app.logger.exception(e)
    return jsonify(error=f"서버 오류: {type(e).__name__}: {e}"), 500


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
    student_no = re.sub(r"\s+", "", request.form.get("student_no", ""))[:20]
    f = request.files.get("audio")
    if not a:
        return jsonify(error="숙제 코드를 찾을 수 없습니다."), 404
    if not student or not f:
        return jsonify(error="이름과 녹음이 필요합니다."), 400
    data, mt = f.read(), f.mimetype or ""
    if len(data) > 4_000_000:
        return jsonify(error="녹음이 너무 깁니다. 다시 녹음하세요. / 录音太长，请重新录音。"), 413
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
        "assignment_id": a["id"], "student": student, "student_no": student_no or None,
        "score": score, "transcript": hyp, "audio_path": path})
    return jsonify(transcript=hyp, score=score, marks=mark(a["sentence"], hyp))


@app.get("/api/config")
def config():
    return jsonify(url=SB_URL, key=SB_ANON)


@app.get("/api/me")
def get_me():
    u, p = me()
    if not u:
        return deny()
    return jsonify(email=u.get("email"), profile=p, active=bool(p) and active(p), is_admin=admin_user(u))


@app.post("/api/me")
def save_me():
    u, p = me()
    if not u:
        return deny()
    d = request.get_json(silent=True) or {}
    name = " ".join(str(d.get("name", "")).split())[:40]
    school = " ".join(str(d.get("school", "")).split())[:60]
    if not name or not school:
        return jsonify(error="이름과 학교를 입력하세요."), 400
    if p:
        sb("PATCH", f"/rest/v1/teachers?id=eq.{u['id']}", json={"name": name, "school": school})
    else:
        sb("POST", "/rest/v1/teachers", json={"id": u["id"], "email": u.get("email"), "name": name, "school": school})
    return jsonify(ok=True)


@app.get("/api/my/classes")
def my_classes():
    u, p = me()
    if not p:
        return deny()
    rows = sb("GET", "/rest/v1/classes", params={
        "teacher_id": f"eq.{u['id']}", "select": "id,school,grade,class_no,join_code",
        "order": "grade.asc,class_no.asc"}).json()
    for r in rows:
        if not r.get("join_code"):  # 코드 도입 전에 만든 반
            r["join_code"] = assign_code(r["id"])
    return jsonify(rows)


@app.post("/api/classes")
def create_class():
    u, p = me()
    if not p:
        return deny()
    if not active(p):
        return not_paid()
    d = request.get_json(silent=True) or {}
    try:
        grade, no = int(d.get("grade")), int(d.get("class_no"))
    except (TypeError, ValueError):
        return jsonify(error="학년과 반은 숫자로 입력하세요."), 400
    if not (1 <= grade <= 12 and 1 <= no <= 99):
        return jsonify(error="학년(1~12), 반(1~99)을 확인하세요."), 400
    key = re.sub(r"\s+", "", p["school"]).lower()
    if sb("GET", "/rest/v1/classes", params={"school_key": f"eq.{key}", "grade": f"eq.{grade}",
                                              "class_no": f"eq.{no}", "select": "id"}).json():
        return jsonify(error="이미 등록된 반입니다."), 409
    for _ in range(5):
        try:
            row = sb("POST", "/rest/v1/classes", json={
                "school": p["school"], "school_key": key, "grade": grade, "class_no": no,
                "teacher_id": u["id"], "join_code": new_code()},
                headers={"Prefer": "return=representation"}).json()[0]
            return jsonify(id=row["id"])
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 409:
                raise
    return jsonify(error="반 코드를 만들지 못했습니다. 다시 시도하세요."), 500


@app.post("/api/classes/<int:cid>/code")
def regen_code(cid):  # 코드가 새어 나갔거나 학기가 바뀔 때 새 코드로 교체
    u, p = me()
    if not p:
        return deny()
    if cid not in my_class_ids(u):
        return jsonify(error="내 반이 아닙니다."), 404
    return jsonify(join_code=assign_code(cid))


@app.delete("/api/assignments/<int:aid>")
def delete_assignment(aid):
    u, p = me()
    if not p:
        return deny()
    if not owns(u, aid):
        return jsonify(error="내 과제가 아닙니다."), 404
    drop_assignments([aid])
    return jsonify(ok=True)


@app.delete("/api/classes/<int:cid>")
def delete_class(cid):
    u, p = me()
    if not p:
        return deny()
    if cid not in my_class_ids(u):
        return jsonify(error="내 반이 아닙니다."), 404
    ids = [a["id"] for a in sb("GET", "/rest/v1/assignments", params={
        "class_id": f"eq.{cid}", "select": "id"}).json()]
    drop_assignments(ids)
    sb("DELETE", "/rest/v1/classes", params={"id": f"eq.{cid}"})
    return jsonify(ok=True)


@app.get("/api/join/<code>")
def join_class(code):  # 학생: 반 코드로 입장 (목록 공개 없음)
    code = code.strip().upper()
    rows = [] if not re.fullmatch(r"[A-Z0-9]{6}", code) else sb("GET", "/rest/v1/classes", params={
        "join_code": f"eq.{code}", "teacher_id": "not.is.null", "select": "id,school,grade,class_no"}).json()
    if not rows:
        return jsonify(error="반 코드를 찾을 수 없습니다. / 找不到班级码。"), 404
    c = rows[0]
    asg = sb("GET", "/rest/v1/assignments", params={
        "class_id": f"eq.{c['id']}", "select": "code,title", "order": "created_at.desc"}).json()
    return jsonify(label=f"{c['school']} {c['grade']}학년 {c['class_no']}반", assignments=asg)


@app.post("/api/assignments")
def create_assignment():
    u, p = me()
    if not p:
        return deny()
    if not active(p):
        return not_paid()
    d = request.get_json(silent=True) or {}
    title, sentence = str(d.get("title", "")).strip(), str(d.get("sentence", "")).strip()
    try:
        cid = int(d.get("class_id"))
    except (TypeError, ValueError):
        cid = None
    if not title or not sentence or cid not in my_class_ids(u):
        return jsonify(error="반, 제목, 문장을 확인하세요."), 400
    row = sb("POST", "/rest/v1/assignments",
             json={"code": secrets.token_hex(3).upper(), "title": title, "sentence": sentence, "class_id": cid},
             headers={"Prefer": "return=representation"}).json()[0]
    return jsonify(row)


@app.get("/api/assignments")
def list_assignments():
    u, p = me()
    if not p:
        return deny()
    ids = my_class_ids(u)
    if not ids:
        return jsonify([])
    return jsonify(sb("GET", "/rest/v1/assignments", params={
        "select": "*", "class_id": "in.(" + ",".join(map(str, ids)) + ")", "order": "created_at.desc"}).json())


@app.get("/api/submissions")
def list_submissions():
    u, p = me()
    if not p:
        return deny()
    aid = request.args.get("assignment_id", "")
    if not aid.isdigit() or not owns(u, aid):
        return jsonify([])
    return jsonify(sb("GET", "/rest/v1/submissions", params={
        "select": "id,student_no,student,score,transcript,audio_path,created_at",
        "assignment_id": f"eq.{aid}", "order": "created_at.desc"}).json())


@app.get("/api/audio/<int:sid>")
def get_audio(sid):
    u, p = me()
    if not p:
        return deny()
    rows = sb("GET", "/rest/v1/submissions",
              params={"id": f"eq.{sid}", "select": "audio_path,assignment_id"}).json()
    if not rows or not rows[0]["audio_path"] or not owns(u, rows[0]["assignment_id"]):
        return jsonify(error="저장된 녹음이 없습니다."), 404
    path = rows[0]["audio_path"]
    r = sb("GET", f"/storage/v1/object/recordings/{path}")
    mime = {"mp4": "audio/mp4", "ogg": "audio/ogg"}.get(path.rsplit(".", 1)[-1], "audio/webm")
    return Response(r.content, mimetype=mime)


KST = timezone(timedelta(hours=9))


def add_months(dt, m):
    y, mo = divmod(dt.month - 1 + m, 12)
    year, month = dt.year + y, mo + 1
    return dt.replace(year=year, month=month, day=min(dt.day, calendar.monthrange(year, month)[1]))


@app.get("/api/admin/teachers")
def admin_teachers():
    if not is_admin():
        return jsonify(error="관리자만 볼 수 있습니다."), 403
    ts = sb("GET", "/rest/v1/teachers", params={
        "select": "id,email,name,school,paid_until,memo,created_at", "order": "created_at.desc"}).json()
    cls = sb("GET", "/rest/v1/classes", params={"teacher_id": "not.is.null", "select": "id,teacher_id"}).json()
    asg = sb("GET", "/rest/v1/assignments", params={"select": "class_id"}).json()
    owner = {c["id"]: c["teacher_id"] for c in cls}
    for t in ts:
        t["n_classes"] = sum(1 for c in cls if c["teacher_id"] == t["id"])
        t["n_assignments"] = sum(1 for a in asg if owner.get(a["class_id"]) == t["id"])
    return jsonify(ts)


@app.post("/api/admin/teachers/<tid>")
def admin_update(tid):
    if not is_admin():
        return jsonify(error="관리자만 쓸 수 있습니다."), 403
    if not re.fullmatch(r"[0-9a-f-]{36}", tid):
        return jsonify(error="형식이 올바르지 않습니다."), 400
    d = request.get_json(silent=True) or {}
    act, upd = d.get("action"), {}
    if act == "extend":
        if d.get("months") not in (1, 3, 12):
            return jsonify(error="기간이 올바르지 않습니다."), 400
        rows = sb("GET", "/rest/v1/teachers", params={"id": f"eq.{tid}", "select": "paid_until"}).json()
        if not rows:
            return jsonify(error="선생님을 찾을 수 없습니다."), 404
        base = datetime.now(KST)  # 만료됐거나 처음이면 오늘부터, 사용 중이면 남은 기간 뒤에 더함
        if rows[0]["paid_until"]:
            base = max(base, datetime.fromisoformat(rows[0]["paid_until"].replace("Z", "+00:00")).astimezone(KST))
        upd["paid_until"] = add_months(base, d["months"]).replace(hour=23, minute=59, second=59, microsecond=0).isoformat()
    elif act == "set":
        day = str(d.get("date", ""))
        if day and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            return jsonify(error="날짜 형식이 올바르지 않습니다."), 400
        upd["paid_until"] = f"{day}T23:59:59+09:00" if day else None
    elif act == "stop":
        upd["paid_until"] = datetime.now(KST).isoformat()
    elif act == "memo":
        upd["memo"] = str(d.get("memo", ""))[:300]
    else:
        return jsonify(error="알 수 없는 요청입니다."), 400
    sb("PATCH", f"/rest/v1/teachers?id=eq.{tid}", json=upd)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
