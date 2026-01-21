from fastapi import FastAPI, Query, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from sentence_transformers import SentenceTransformer
import torch
import faiss
import json
import os
import pathlib
import csv
import secrets
import threading
import time
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import re
import numpy as np
from collections import Counter
import unicodedata

# ============================================
# 設定
# ============================================

APP_TITLE = "サポート検索（動画 + FAQ）"
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)

DEFAULT_TOP_K = 10
DEFAULT_PAGE_LIMIT = 10
MAX_PAGE_LIMIT = 50

# 起動時の重い初期化（大きいJSON読込・FAISS構築・モデルロード）を
# デプロイ/ヘルスチェックのタイムアウトを避けるためバックグラウンドで実行する
INIT_STATE = {
    "started": False,
    "ready": False,
    "error": None,
    "stage": "not_started",  # not_started|loading_videos|building_video_index|loading_faq|building_faq_index|ready|error
    "started_at": None,
    "updated_at": None,
    "eta_seconds": None,
}
_INIT_LOCK = threading.Lock()

# 管理者（Basic認証）
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "changeme")  # 本番では必ず変更

BASE_DIR = pathlib.Path(__file__).parent
DATA_PATH = BASE_DIR / "data.json"
SYNONYMS_PATH = BASE_DIR / "synonyms.json"
FAQ_PATH = BASE_DIR / "faq_chatbot_fixed_only.json"
SEARCH_LOG_PATH = BASE_DIR / "search_logs.csv"

# ============================================
# アプリ
# ============================================
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "abc123"


app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

# 動画検索用
videos: List[Dict[str, Any]] = []
text_corpus: List[str] = []
synonyms: Dict[str, List[str]] = {}

_model: Optional[SentenceTransformer] = None
video_index: Optional[faiss.IndexFlatIP] = None  # cosine (normalized) via inner product

# FAQ検索用
faq_data: Dict[str, Any] = {}
faq_items_flat: List[Dict[str, Any]] = []
faq_corpus: List[str] = []
faq_index: Optional[faiss.IndexFlatIP] = None


# ============================================
# 共通ユーティリティ
# ============================================

def normalize_text(text: str) -> str:
    """
    精度向上のための正規化
    - NFKC（全角/半角・記号ゆれ統一）
    - lower
    - 連続スペースを1つ
    """
    text = (text or "")
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USER)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASS)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


def safe_load_json(path: pathlib.Path, default: Any):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{path.name} の読み込みに失敗: {e}")


def safe_write_json(path: pathlib.Path, obj: Any):
    try:
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{path.name} の保存に失敗: {e}")


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return _model


def normalize_embeddings(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype("float32")


# ============================================
# 類義語展開
# ============================================

def expand_query_with_synonyms(query: str) -> str:
    """
    単語でも短文でもOKな類義語展開:
    - トークン一致で展開
    - クエリ文字列に key が含まれていれば部分一致でも展開
    - 重複を順序維持で除去
    """
    if not synonyms:
        return query

    original_query = query
    tokens = list(filter(None, re.split(r"\s+", original_query)))

    expanded: List[str] = []

    for t in tokens:
        expanded.append(t)
        if t in synonyms:
            expanded.extend([normalize_text(s) for s in synonyms[t]])

    for key, syns in synonyms.items():
        k = normalize_text(key)
        if k and k in original_query and k not in tokens:
            expanded.append(k)
            expanded.extend([normalize_text(s) for s in syns])

    seen = set()
    unique = []
    for w in expanded:
        w = normalize_text(w)
        if w and w not in seen:
            seen.add(w)
            unique.append(w)

    return " ".join(unique) if unique else query


# ============================================
# 動画データ読み込み & インデックス
# ============================================

def load_videos() -> None:
    global videos, synonyms, text_corpus

    if not DATA_PATH.exists():
        raise RuntimeError(f"data.json が見つかりません: {DATA_PATH}")

    videos_loaded = safe_load_json(DATA_PATH, default=[])
    if not isinstance(videos_loaded, list):
        raise RuntimeError("data.json の形式が不正です（listを期待）")
    videos = videos_loaded

    synonyms_loaded = safe_load_json(SYNONYMS_PATH, default={})
    synonyms = synonyms_loaded if isinstance(synonyms_loaded, dict) else {}

    text_corpus = []
    for v in videos:
        title = normalize_text(v.get("title", ""))
        desc = normalize_text(v.get("description", ""))
        trans = normalize_text(v.get("transcript", ""))

        combined = f"{title} [SEP] {desc} [SEP] {trans}"
        # title を強める
        combined_weighted = f"{title} {title} {title} {combined}"
        text_corpus.append(combined_weighted)


def build_video_index() -> None:
    global video_index
    if not text_corpus:
        video_index = None
        return

    m = get_model()
    emb = m.encode(text_corpus, convert_to_numpy=True, show_progress_bar=False)
    emb = normalize_embeddings(emb)

    dim = emb.shape[1]
    video_index = faiss.IndexFlatIP(dim)
    video_index.add(emb)


def search_videos_ranked(query: str, k: int) -> List[Dict[str, Any]]:
    """
    k件分の候補（スコア付き）を返す（内部）
    """
    if video_index is None or not videos:
        return []

    q_norm = normalize_text(query)
    q_for_embed = expand_query_with_synonyms(q_norm)

    m = get_model()
    q_emb = m.encode([q_for_embed], convert_to_numpy=True, show_progress_bar=False)
    q_emb = normalize_embeddings(q_emb)

    sims, idxs = video_index.search(q_emb, min(k, len(videos)))
    sims = sims[0]
    idxs = idxs[0]

    query_tokens = set(q_norm.split())
    results: List[Dict[str, Any]] = []

    for sim, idx in zip(sims, idxs):
        if idx < 0 or idx >= len(videos):
            continue
        v = dict(videos[idx])

        # キーワード一致（title/description）を軽くブースト
        text_for_kw = normalize_text((v.get("title", "") or "") + " " + (v.get("description", "") or ""))
        kw_score = 0.0
        if query_tokens:
            hit = sum(1 for t in query_tokens if t and t in text_for_kw)
            kw_score = hit / len(query_tokens)

        v["_score"] = 0.9 * float(sim) + 0.1 * kw_score
        results.append(v)

    results.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    return results


# ============================================
# FAQ読み込み & インデックス（meta + faqs 形式対応）
# ============================================

def load_faq_raw() -> Dict[str, Any]:
    obj = safe_load_json(FAQ_PATH, default={"meta": {}, "faqs": []})
    if isinstance(obj, list):
        # 過去形式（list）の場合はラップ
        obj = {"meta": {}, "faqs": obj}
    if not isinstance(obj, dict):
        obj = {"meta": {}, "faqs": []}
    if "faqs" not in obj or not isinstance(obj["faqs"], list):
        obj["faqs"] = []
    if "meta" not in obj or not isinstance(obj["meta"], dict):
        obj["meta"] = {}
    return obj


def save_faq_raw(obj: Dict[str, Any]) -> None:
    # 形式を守る
    if "faqs" not in obj or not isinstance(obj["faqs"], list):
        raise HTTPException(status_code=400, detail="FAQの形式が不正です（faqsが必要）")
    if "meta" not in obj or not isinstance(obj["meta"], dict):
        obj["meta"] = {}
    safe_write_json(FAQ_PATH, obj)


def flatten_faq(obj: Any) -> List[Dict[str, Any]]:
    """
    {"meta":..., "faqs":[...]} を前提に、配列へ正規化
    """
    if isinstance(obj, dict) and "faqs" in obj and isinstance(obj["faqs"], list):
        obj = obj["faqs"]

    items: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        for i, it in enumerate(obj):
            if isinstance(it, dict):
                it2 = dict(it)
                it2["_key"] = str(it2.get("id", i))
                items.append(it2)
            else:
                items.append({"_key": str(i), "raw": it})
    return items


def faq_item_to_text(item: Dict[str, Any]) -> str:
    """
    重み付け：
    - question/utterances/keywords を強く
    - steps は長文化しやすいので先頭2行のみ
    """
    q = normalize_text(item.get("question", "") or "")
    category = normalize_text(item.get("category", "") or "")
    intent = normalize_text(item.get("intent", "") or "")

    utter = item.get("utterances", [])
    utter_text = " ".join([normalize_text(u) for u in utter]) if isinstance(utter, list) else normalize_text(str(utter))

    kws = item.get("keywords", [])
    kw_list = [normalize_text(k) for k in kws] if isinstance(kws, list) else [normalize_text(str(kws))]
    kw_text = " ".join([k for k in kw_list if k])

    steps = item.get("steps", [])
    if isinstance(steps, list):
        steps_head = steps[:2]
        steps_text = " ".join([normalize_text(s) for s in steps_head if s])
    else:
        steps_text = normalize_text(str(steps))[:200]

    parts = []
    if q:
        parts += [q, q, q]
    if utter_text:
        parts += [utter_text, utter_text]
    if kw_text:
        parts += [kw_text, kw_text, kw_text]
    if category:
        parts.append(category)
    if intent:
        parts.append(intent)
    if steps_text:
        parts.append(steps_text)

    return " [SEP] ".join([p for p in parts if p])


def load_faq() -> None:
    global faq_data, faq_items_flat, faq_corpus
    faq_data = load_faq_raw()
    faq_items_flat = flatten_faq(faq_data)
    faq_corpus = [faq_item_to_text(it) for it in faq_items_flat]


def build_faq_index() -> None:
    global faq_index
    if not faq_corpus:
        faq_index = None
        return

    m = get_model()
    emb = m.encode(faq_corpus, convert_to_numpy=True, show_progress_bar=False)
    emb = normalize_embeddings(emb)

    dim = emb.shape[1]
    faq_index = faiss.IndexFlatIP(dim)
    faq_index.add(emb)


def faq_keyword_boost(query_norm: str, item: Dict[str, Any]) -> float:
    """
    query と FAQ の keywords / question / utterances の一致でブースト
    """
    score = 0.0
    q = query_norm

    kws = item.get("keywords", [])
    kw_list = kws if isinstance(kws, list) else [kws]
    for k in kw_list:
        k2 = normalize_text(str(k))
        if k2 and (k2 in q or q in k2):
            score += 1.2

    question = normalize_text(item.get("question", "") or "")
    if question and (q in question or question in q):
        score += 0.8

    utter = item.get("utterances", [])
    if isinstance(utter, list):
        for u in utter:
            u2 = normalize_text(str(u))
            if u2 and (u2 in q or q in u2):
                score += 0.4

    return score


def search_faq_ranked(query: str, k: int) -> List[Dict[str, Any]]:
    if faq_index is None or not faq_items_flat:
        return []

    q_norm = normalize_text(query)
    q_for_embed = expand_query_with_synonyms(q_norm)

    m = get_model()
    q_emb = m.encode([q_for_embed], convert_to_numpy=True, show_progress_bar=False)
    q_emb = normalize_embeddings(q_emb)

    # まず多めに候補を取って再ランキング
    k_search = min(max(k, DEFAULT_TOP_K) * 5, len(faq_items_flat))
    sims, idxs = faq_index.search(q_emb, k_search)
    sims = sims[0]
    idxs = idxs[0]

    results: List[Dict[str, Any]] = []
    for sim, idx in zip(sims, idxs):
        if idx < 0 or idx >= len(faq_items_flat):
            continue
        it = dict(faq_items_flat[idx])

        vec = float(sim)
        boost = faq_keyword_boost(q_norm, it)
        it["_score"] = 0.85 * vec + 0.15 * boost
        results.append(it)

    results.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    return results[:k]


# ============================================
# 検索ログ
# ============================================

def log_search_query(query: str, hits_count: int) -> None:
    header = ["timestamp", "query", "hits"]
    now = datetime.now(timezone.utc).isoformat()
    row = [now, query, str(hits_count)]
    file_exists = SEARCH_LOG_PATH.exists()

    with open(SEARCH_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)


def parse_logs() -> List[dict]:
    if not SEARCH_LOG_PATH.exists():
        return []
    rows = []
    with open(SEARCH_LOG_PATH, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ts = row.get("timestamp", "")
            q = row.get("query", "")
            hits = int(row.get("hits", "0") or 0)
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except:
                continue
            rows.append({"dt": dt, "query": q, "hits": hits})
    return rows


# ============================================
# 起動時
# ===================================
def _set_init_state(stage: str, eta_seconds: int | None = None) -> None:
    INIT_STATE["stage"] = stage
    INIT_STATE["updated_at"] = time.time()
    if eta_seconds is not None:
        INIT_STATE["eta_seconds"] = int(eta_seconds)



def _background_initialize() -> None:
    """重い初期化をバックグラウンドで実行する（デプロイ時タイムアウト対策）"""
    with _INIT_LOCK:
        if INIT_STATE["started"] and not INIT_STATE.get("error"):
            return
        INIT_STATE["started"] = True
        INIT_STATE["ready"] = False
        INIT_STATE["error"] = None
        INIT_STATE["started_at"] = time.time()
        INIT_STATE["updated_at"] = INIT_STATE["started_at"]
        INIT_STATE["eta_seconds"] = 180  # 目安（環境・データ量で変動）

    try:
        _set_init_state("loading_videos", 150)
        load_videos()

        _set_init_state("building_video_index", 120)
        build_video_index()

        _set_init_state("loading_faq", 60)
        load_faq()

        _set_init_state("building_faq_index", 30)
        build_faq_index()

        INIT_STATE["ready"] = True
        _set_init_state("ready", 0)
    except Exception as e:
        INIT_STATE["error"] = f"{type(e).__name__}: {e}"
        INIT_STATE["ready"] = False
        _set_init_state("error", None)
        # 失敗した場合でもリトライできるように started は True のまま残すが、
        # 次回アクセス時に _background_initialize が再実行されるよう error を見て分岐する

def _ensure_ready() -> None:
    """検索/管理API実行前に初期化状態を確認する"""
    if INIT_STATE.get("error"):
        raise HTTPException(status_code=500, detail=f"Initialization failed: {INIT_STATE['error']}")
    if not INIT_STATE.get("ready"):
        # 初回アクセス時に初期化が開始されていないケースに備えてキック
        threading.Thread(target=_background_initialize, daemon=True).start()
        # フロントが落ちないよう 503 ではなく空結果を返せるように呼び出し元で分岐する
        raise HTTPException(status_code=503, detail="Index is warming up. Please retry in a moment.")




@app.on_event("startup")
def on_startup() -> None:
    # デプロイ/ヘルスチェックのタイムアウトを避けるため、初期化はバックグラウンドで行う
    threading.Thread(target=_background_initialize, daemon=True).start()



# ============================================
# 初期化ステータス（ウォームアップ表示用）
# ============================================
@app.get("/init/status")
def init_status() -> dict:
    started_at = INIT_STATE.get("started_at")
    now = time.time()
    elapsed = int(now - started_at) if started_at else 0
    eta = INIT_STATE.get("eta_seconds")
    remaining = max(0, int(eta - elapsed)) if isinstance(eta, int) else None
    return {
        "started": INIT_STATE.get("started", False),
        "ready": INIT_STATE.get("ready", False),
        "stage": INIT_STATE.get("stage"),
        "error": INIT_STATE.get("error"),
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "remaining_seconds": remaining,
    }

# ============================================
# 検索API（ページング対応）
#   - paged=1 のとき {items, has_more, offset, limit, total_visible} を返す
#   - paged=0 のとき互換のため array を返す（top_kのみ）
# ============================================

@app.get("/search", summary="動画検索", tags=["search"])
def search_videos_endpoint(
    query: str | None = Query(None),
    q: str | None = Query(None),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=200),
    paged: int = Query(0, ge=0, le=1),
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
):
    # 初期化が完了していない場合は空結果（UIが落ちない）で返す
    try:
        _ensure_ready()
    except HTTPException as e:
        if e.status_code == 503:
            return {"items": [], "offset": offset, "limit": limit, "has_more": False, "total_visible": 0, "warming_up": True, "init_error": INIT_STATE.get("error")} if paged == 1 else []
        raise

    query = (query or q or "").strip()
    if not query:
        return {"items": [], "offset": offset, "limit": limit, "has_more": False, "total_visible": 0} if paged == 1 else []
    if paged == 0:
        results = search_videos_ranked(query, top_k)
        log_search_query(query, len(results))
        # UIの都合で transcript は返すが、フロント側で非表示にしている
        return results

    # ページング: offset+limit+1 件まで取得して has_more 判定
    need = min(offset + limit + 1, 500)  # 過剰取得を抑制（必要なら上げる）
    ranked = search_videos_ranked(query, need)
    items = ranked[offset: offset + limit]
    has_more = len(ranked) > offset + limit

    log_search_query(query, len(items))
    return {
        "items": items,
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
        "total_visible": len(ranked),  # 取得できた範囲での可視総数（近似）
    }


@app.get("/faq/search", summary="FAQ検索", tags=["faq"])
def search_faq_endpoint(
    query: str | None = Query(None),
    q: str | None = Query(None),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=200),
    paged: int = Query(0, ge=0, le=1),
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
):
    # 初期化が完了していない場合は空結果（UIが落ちない）で返す
    try:
        _ensure_ready()
    except HTTPException as e:
        if e.status_code == 503:
            return {"items": [], "offset": offset, "limit": limit, "has_more": False, "total_visible": 0, "warming_up": True, "init_error": INIT_STATE.get("error")} if paged == 1 else []
        raise

    query = (query or q or "").strip()
    if not query:
        return {"items": [], "offset": offset, "limit": limit, "has_more": False, "total_visible": 0} if paged == 1 else []
    if paged == 0:
        results = search_faq_ranked(query, top_k)
        return results

    need = min(offset + limit + 1, 500)
    ranked = search_faq_ranked(query, need)
    items = ranked[offset: offset + limit]
    has_more = len(ranked) > offset + limit

    return {
        "items": items,
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
        "total_visible": len(ranked),
    }


@app.get("/faq/debug", tags=["faq"])
def faq_debug():
    return {
        "faq_file_exists": FAQ_PATH.exists(),
        "faq_items_count": len(faq_items_flat),
        "faq_index_ready": faq_index is not None,
        "sample": faq_items_flat[0] if faq_items_flat else None,
    }


# ============================================
# 管理API: synonyms.json CRUD + 一括生成
# ============================================

@app.get("/admin/api/synonyms", tags=["admin"])
def admin_get_synonyms(user: str = Depends(verify_admin)):
    return safe_load_json(SYNONYMS_PATH, default={})


@app.put("/admin/api/synonyms", tags=["admin"])
def admin_put_synonyms(payload: Dict[str, List[str]] = Body(...), user: str = Depends(verify_admin)):
    safe_write_json(SYNONYMS_PATH, payload)
    global synonyms
    synonyms = payload if isinstance(payload, dict) else {}
    return {"ok": True}


@app.patch("/admin/api/synonyms/{term}", tags=["admin"])
def admin_patch_synonym_term(
    term: str,
    values: List[str] = Body(..., description="この term の類義語リスト"),
    user: str = Depends(verify_admin),
):
    current = safe_load_json(SYNONYMS_PATH, default={})
    if not isinstance(current, dict):
        current = {}
    current[term] = values
    safe_write_json(SYNONYMS_PATH, current)
    global synonyms
    synonyms = current
    return {"ok": True, "term": term, "values": values}


@app.delete("/admin/api/synonyms/{term}", tags=["admin"])
def admin_delete_synonym_term(term: str, user: str = Depends(verify_admin)):
    current = safe_load_json(SYNONYMS_PATH, default={})
    if isinstance(current, dict) and term in current:
        del current[term]
        safe_write_json(SYNONYMS_PATH, current)
        global synonyms
        synonyms = current
    return {"ok": True, "deleted": term}


def generate_synonyms_from_data(videos_: List[Dict[str, Any]], faq_items_: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    data.json + FAQ keywords から簡易的に “表記ゆれ” シノニムを生成
    例：DXF / dxf / ＤＸＦ / DXFデータ など
    """
    def norm(s: str) -> str:
        return unicodedata.normalize("NFKC", s or "").strip()

    cand: List[str] = []

    # FAQ keywords
    for f in faq_items_:
        kws = f.get("keywords", [])
        if isinstance(kws, list):
            cand += [norm(str(x)) for x in kws if str(x).strip()]

    # data.json title/description
    for v in videos_:
        t = norm(v.get("title",""))
        d = norm(v.get("description",""))
        text = f"{t} {d}"
        cand += re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\_]{1,30}", text)  # DXF, SQLServer2022
        cand += re.findall(r"[ァ-ヴー]{3,20}", text)  # カタカナ語

    c = Counter([x for x in cand if x])
    vocab = [w for w, n in c.items() if n >= 2]

    syn: Dict[str, List[str]] = {}
    for w in vocab:
        base = norm(w)
        if not base:
            continue

        variants = set()
        variants.add(base)
        variants.add(base.lower())
        variants.add(base.upper())
        variants.add(base.replace(" ", ""))

        # ありがちな表記ゆれ（必要に応じて増やす）
        variants.add(base.replace("II", "Ⅱ"))
        variants.add(base.replace("ⅱ", "Ⅱ"))
        variants.add(base.replace("２", "2"))

        vals = sorted(set(v for v in variants if v) - {base})
        if vals:
            syn[normalize_text(base)] = [normalize_text(v) for v in vals if normalize_text(v) != normalize_text(base)]

    return syn


@app.post("/admin/api/synonyms/generate", tags=["admin"])
def admin_generate_synonyms(user: str = Depends(verify_admin)):
    gen = generate_synonyms_from_data(videos, faq_items_flat)
    safe_write_json(SYNONYMS_PATH, gen)
    global synonyms
    synonyms = gen
    return {"ok": True, "count": len(gen)}


# ============================================
# 管理API: FAQ（meta + faqs 形式） CRUD
#  - 全体取得/保存
#  - 1件単位（idキー）
#  - 一覧（フィルタ + offset/limit）
# ============================================

@app.get("/admin/api/faq", tags=["admin"])
def admin_get_faq(user: str = Depends(verify_admin)):
    return load_faq_raw()


@app.put("/admin/api/faq", tags=["admin"])
def admin_put_faq(payload: Dict[str, Any] = Body(...), user: str = Depends(verify_admin)):
    # 形式を整えて保存
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="FAQは object 形式が必要です（meta+faqs）")
    if "faqs" not in payload or not isinstance(payload["faqs"], list):
        raise HTTPException(status_code=400, detail="FAQは faqs(list) を含む必要があります")
    if "meta" not in payload or not isinstance(payload["meta"], dict):
        payload["meta"] = {}
    save_faq_raw(payload)

    # 即反映
    load_faq()
    build_faq_index()
    return {"ok": True, "count": len(payload["faqs"])}


def find_faq_index_by_id(faqs: List[Dict[str, Any]], faq_id: str) -> int:
    for i, it in enumerate(faqs):
        if str(it.get("id", "")).strip() == faq_id:
            return i
    return -1


@app.get("/admin/api/faq/items", tags=["admin"])
def admin_list_faq_items(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    q: str = Query("", description="filter（question/keywords/category）"),
    user: str = Depends(verify_admin),
):
    _ensure_ready()

    obj = load_faq_raw()
    faqs = obj.get("faqs", [])
    if not isinstance(faqs, list):
        faqs = []

    qn = normalize_text(q)
    if qn:
        def hit(it: Dict[str, Any]) -> bool:
            t = " ".join([
                str(it.get("question","")),
                str(it.get("category","")),
                " ".join(it.get("keywords", []) if isinstance(it.get("keywords", []), list) else [str(it.get("keywords",""))]),
            ])
            return qn in normalize_text(t)
        faqs = [it for it in faqs if isinstance(it, dict) and hit(it)]
    else:
        faqs = [it for it in faqs if isinstance(it, dict)]

    # id順に安定化（FAQ-0001 形式を想定）
    def sort_key(it: Dict[str, Any]):
        s = str(it.get("id",""))
        m = re.search(r"(\d+)$", s)
        return (s[:4], int(m.group(1)) if m else 10**9, s)
    faqs.sort(key=sort_key)

    items = faqs[offset: offset + limit]
    has_more = len(faqs) > offset + limit
    return {"items": items, "offset": offset, "limit": limit, "has_more": has_more, "total": len(faqs)}


@app.get("/admin/api/faq/item/{faq_id}", tags=["admin"])
def admin_get_faq_item(faq_id: str, user: str = Depends(verify_admin)):
    _ensure_ready()

    obj = load_faq_raw()
    faqs = obj.get("faqs", [])
    if not isinstance(faqs, list):
        faqs = []
    idx = find_faq_index_by_id(faqs, faq_id)
    if idx < 0:
        raise HTTPException(status_code=404, detail="FAQが見つかりません")
    return faqs[idx]


@app.post("/admin/api/faq/item", tags=["admin"])
def admin_create_faq_item(item: Dict[str, Any] = Body(...), user: str = Depends(verify_admin)):
    _ensure_ready()

    faq_id = str(item.get("id", "")).strip()
    if not faq_id:
        raise HTTPException(status_code=400, detail="id が必要です")

    obj = load_faq_raw()
    faqs = obj.get("faqs", [])
    if not isinstance(faqs, list):
        faqs = []

    if find_faq_index_by_id(faqs, faq_id) >= 0:
        raise HTTPException(status_code=409, detail="同じidが既に存在します")

    faqs.append(item)
    obj["faqs"] = faqs
    save_faq_raw(obj)

    load_faq()
    build_faq_index()
    return {"ok": True, "id": faq_id}


@app.patch("/admin/api/faq/item/{faq_id}", tags=["admin"])
def admin_update_faq_item(faq_id: str, item: Dict[str, Any] = Body(...), user: str = Depends(verify_admin)):
    _ensure_ready()

    faq_id = str(faq_id).strip()
    if not faq_id:
        raise HTTPException(status_code=400, detail="id が不正です")

    obj = load_faq_raw()
    faqs = obj.get("faqs", [])
    if not isinstance(faqs, list):
        faqs = []

    idx = find_faq_index_by_id(faqs, faq_id)
    if idx < 0:
        # ない場合は追加（運用上便利）
        item["id"] = faq_id
        faqs.append(item)
        obj["faqs"] = faqs
        save_faq_raw(obj)
        load_faq()
        build_faq_index()
        return {"ok": True, "id": faq_id, "created": True}

    item["id"] = faq_id
    faqs[idx] = item
    obj["faqs"] = faqs
    save_faq_raw(obj)

    load_faq()
    build_faq_index()
    return {"ok": True, "id": faq_id, "created": False}


@app.delete("/admin/api/faq/item/{faq_id}", tags=["admin"])
def admin_delete_faq_item(faq_id: str, user: str = Depends(verify_admin)):
    _ensure_ready()

    obj = load_faq_raw()
    faqs = obj.get("faqs", [])
    if not isinstance(faqs, list):
        faqs = []

    idx = find_faq_index_by_id(faqs, str(faq_id).strip())
    if idx < 0:
        raise HTTPException(status_code=404, detail="FAQが見つかりません")

    faqs.pop(idx)
    obj["faqs"] = faqs
    save_faq_raw(obj)

    load_faq()
    build_faq_index()
    return {"ok": True, "deleted": faq_id}


# ============================================
# 管理API: 検索ログ集計（月別/日別）
# ============================================

@app.get("/admin/api/logs/months", tags=["admin"])
def admin_list_log_months(user: str = Depends(verify_admin)):
    rows = parse_logs()
    months = sorted({r["dt"].strftime("%Y-%m") for r in rows}, reverse=True)
    return {"months": months}


@app.get("/admin/api/logs/summary", tags=["admin"])
def admin_logs_summary(
    month: str = Query(..., description="YYYY-MM"),
    user: str = Depends(verify_admin),
):
    rows = parse_logs()
    day_counter = Counter()
    query_counter = Counter()

    for r in rows:
        if r["dt"].strftime("%Y-%m") != month:
            continue
        day = r["dt"].strftime("%Y-%m-%d")
        day_counter[day] += 1
        query_counter[r["query"]] += 1

    days = [{"day": d, "count": c} for d, c in sorted(day_counter.items())]
    top_queries = [{"query": q, "count": c} for q, c in query_counter.most_common(50)]
    return {"month": month, "days": days, "top_queries": top_queries}


@app.get("/admin/api/logs/export", tags=["admin"])
def admin_export_logs_csv(user: str = Depends(verify_admin)):
    if not SEARCH_LOG_PATH.exists():
        raise HTTPException(status_code=404, detail="検索ログがまだありません")

    csv_data = SEARCH_LOG_PATH.read_text(encoding="utf-8")
    headers = {"Content-Disposition": 'attachment; filename="search_logs.csv"'}
    return StreamingResponse(iter([csv_data]), media_type="text/csv", headers=headers)


# ============================================
# 画面配信（frontend + admin_ui）
# ============================================

frontend_path = BASE_DIR / "frontend"
admin_path = BASE_DIR / "admin_ui"

if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

admin_path.mkdir(parents=True, exist_ok=True)
app.mount("/admin/static", StaticFiles(directory=admin_path), name="admin_static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_index():
    index_file = frontend_path / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return index_file.read_text(encoding="utf-8")


@app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_home():
    f = admin_path / "index.html"
    if not f.exists():
        return HTMLResponse("<h1>admin index.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")


@app.get("/admin/logs", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_logs():
    f = admin_path / "logs.html"
    if not f.exists():
        return HTMLResponse("<h1>admin logs.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")


@app.get("/admin/faq", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_faq():
    f = admin_path / "faq.html"
    if not f.exists():
        return HTMLResponse("<h1>admin faq.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")
