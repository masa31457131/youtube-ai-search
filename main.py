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
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import re
import numpy as np
from collections import Counter

# ============================================
# 設定
# ============================================

APP_TITLE = "音声検索AI - サポート検索 (高速・高精度版)"
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
DEFAULT_TOP_K = 10

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

model: Optional[SentenceTransformer] = None
index: Optional[faiss.IndexFlatIP] = None  # cosine (normalized) via inner product

# FAQ検索用
faq_data: Any = None
faq_items_flat: List[Dict[str, Any]] = []
faq_corpus: List[str] = []
faq_index: Optional[faiss.IndexFlatIP] = None


# ============================================
# 共通ユーティリティ
# ============================================

import unicodedata

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


# ============================================
# 動画データ読み込み & インデックス
# ============================================

def load_data() -> None:
    global videos, synonyms, text_corpus

    if not DATA_PATH.exists():
        raise RuntimeError(f"data.json が見つかりません: {DATA_PATH}")

    videos = safe_load_json(DATA_PATH, default=[])

    synonyms_loaded = safe_load_json(SYNONYMS_PATH, default={})
    synonyms = synonyms_loaded if isinstance(synonyms_loaded, dict) else {}

    text_corpus = []
    for v in videos:
        title = normalize_text(v.get("title", ""))
        desc = normalize_text(v.get("description", ""))
        trans = normalize_text(v.get("transcript", ""))

        combined = f"{title} [SEP] {desc} [SEP] {trans}"
        combined_weighted = f"{title} {title} {combined}"  # title boost
        text_corpus.append(combined_weighted)


def get_model() -> SentenceTransformer:
    global model
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return model


def build_search_index() -> None:
    global index
    if not text_corpus:
        index = None
        return

    m = get_model()
    emb = m.encode(text_corpus, convert_to_numpy=True, show_progress_bar=False)

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = (emb / norms).astype("float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)


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
            expanded.extend(synonyms[t])

    for key, syns in synonyms.items():
        if key in original_query and key not in tokens:
            expanded.append(key)
            expanded.extend(syns)

    seen = set()
    unique = []
    for w in expanded:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    return " ".join(unique)


def search_core(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    if index is None or not videos:
        return []

    m = get_model()
    q_norm = normalize_text(query)
    q_for_embed = expand_query_with_synonyms(q_norm)

    q_emb = m.encode([q_for_embed], convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    q_emb = (q_emb / norms).astype("float32")

    k = min(top_k * 5, len(videos))
    sims, idxs = index.search(q_emb, k)
    sims = sims[0]
    idxs = idxs[0]

    query_tokens = set(q_norm.split())
    results: List[tuple[float, Dict[str, Any]]] = []

    for sim, idx in zip(sims, idxs):
        if idx < 0 or idx >= len(videos):
            continue
        v = videos[idx]

        text_for_kw = normalize_text((v.get("title", "") or "") + " " + (v.get("description", "") or ""))
        kw_score = 0.0
        if query_tokens:
            hit = sum(1 for t in query_tokens if t and t in text_for_kw)
            kw_score = hit / len(query_tokens)

        final_score = 0.85 * float(sim) + 0.15 * kw_score
        results.append((final_score, v))

    results.sort(key=lambda x: x[0], reverse=True)
    return [v for _, v in results[:top_k]]


# ============================================
# FAQ読み込み & インデックス
# ============================================

def flatten_faq(obj: Any) -> List[Dict[str, Any]]:
    """
    FAQ json の代表的な形式をすべて配列に正規化する
    対応：
      - {"faqs":[...], "meta":{...}}  ← あなたの形式
      - {"items":[...]}
      - [...](list)
      - { "id": {...}, ... } (dict)
    """
    # まず「ラッパー」形式を吸収
    if isinstance(obj, dict):
        if "faqs" in obj and isinstance(obj["faqs"], list):
            obj = obj["faqs"]
        elif "items" in obj and isinstance(obj["items"], list):
            obj = obj["items"]

    items: List[Dict[str, Any]] = []

    if isinstance(obj, list):
        for i, it in enumerate(obj):
            if isinstance(it, dict):
                it2 = dict(it)
                # id があればそれを key に
                it2["_key"] = str(it2.get("id", i))
                items.append(it2)
            else:
                items.append({"_key": str(i), "raw": it})
        return items

    if isinstance(obj, dict):
        # dict の場合（id->item 形式など）
        for k, it in obj.items():
            if isinstance(it, dict):
                it2 = dict(it)
                it2["_key"] = str(it2.get("id", k))
                items.append(it2)
            else:
                items.append({"_key": str(k), "raw": it})
        return items

    return items


def faq_item_to_text(item: Dict[str, Any]) -> str:
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
        # 長文に引っ張られないよう先頭だけ（必要なら 2〜3 行に増やす）
        steps_head = steps[:2]
        steps_text = " ".join([normalize_text(s) for s in steps_head if s])
    else:
        steps_text = normalize_text(str(steps))[:200]  # 一応制限

    parts = []
    if q:
        parts += [q, q, q]                  # 質問を強く
    if utter_text:
        parts += [utter_text, utter_text]   # 言い回しも強め
    if kw_text:
        parts += [kw_text, kw_text, kw_text] # キーワード最重要
    if category:
        parts.append(category)
    if intent:
        parts.append(intent)
    if steps_text:
        parts.append(steps_text)

    return " [SEP] ".join([p for p in parts if p])
    

def faq_keyword_boost(query_norm: str, item: Dict[str, Any]) -> float:
    """
    query と FAQ の keywords / question / utterances の一致でブースト
    """
    score = 0.0
    q = query_norm

    # keywordsに含まれる語がクエリに含まれていれば強く加点
    kws = item.get("keywords", [])
    kw_list = kws if isinstance(kws, list) else [kws]
    for k in kw_list:
        k2 = normalize_text(str(k))
        if k2 and (k2 in q or q in k2):
            score += 1.2  # 強め

    # question/utterances の部分一致
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


def load_faq() -> None:
    global faq_data, faq_items_flat, faq_corpus
    faq_data = safe_load_json(FAQ_PATH, default=[])
    faq_items_flat = flatten_faq(faq_data)
    faq_corpus = [faq_item_to_text(it) for it in faq_items_flat]


def build_faq_index() -> None:
    global faq_index
    if not faq_corpus:
        faq_index = None
        return

    m = get_model()
    emb = m.encode(faq_corpus, convert_to_numpy=True, show_progress_bar=False)

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = (emb / norms).astype("float32")

    dim = emb.shape[1]
    faq_index = faiss.IndexFlatIP(dim)
    faq_index.add(emb)


def search_faq_core(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    if faq_index is None or not faq_items_flat:
        return []

    m = get_model()
    q_norm = normalize_text(query)
    q_for_embed = expand_query_with_synonyms(q_norm)

    q_emb = m.encode([q_for_embed], convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    q_emb = (q_emb / norms).astype("float32")

    k = min(top_k * 5, len(faq_items_flat))
    sims, idxs = faq_index.search(q_emb, k)
    sims = sims[0]
    idxs = idxs[0]

    results = []
    for sim, idx in zip(sims, idxs):
        if idx < 0 or idx >= len(faq_items_flat):
            continue
        it = dict(faq_items_flat[idx])

        # ベクトルスコア
        vec = float(sim)

        # 文字一致ブースト
        boost = faq_keyword_boost(q_norm, it)

        # ハイブリッド最終スコア（重みはここがチューニング点）
        it["_score"] = 0.85 * vec + 0.15 * boost

        results.append(it)

    results.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
    return results[:top_k]


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
# ============================================

@app.on_event("startup")
def on_startup() -> None:
    load_data()
    build_search_index()
    load_faq()
    build_faq_index()


# ============================================
# 検索API（フロント互換）
#   - index.html は /search と /faq/search を別々に呼んでいる
# ============================================

@app.get("/search", summary="動画検索（配列で返す）", tags=["search"])
def search_endpoint(
    query: str = Query(..., min_length=1, description="検索クエリ（単語・短文OK）"),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=50, description="返す件数"),
):
    results = search_core(query, top_k=top_k)
    # FAQは別APIで取るので、ここでは動画件数だけログに入れる（必要なら合算も可）
    log_search_query(query, len(results))
    return results


@app.get("/faq/search", summary="FAQ検索（配列で返す）", tags=["faq"])
def faq_search_endpoint(
    query: str = Query(..., min_length=1, description="検索クエリ（単語・短文OK）"),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=50, description="返す件数"),
):
    results = search_faq_core(query, top_k=top_k)
    return results


# ============================================
# 管理API: synonyms.json CRUD
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


# ============================================
# 管理API: FAQ CRUD（保存後に FAQ index 再構築）
# ============================================

def faq_load():
    return safe_load_json(FAQ_PATH, default=[])


def faq_save(obj: Any):
    safe_write_json(FAQ_PATH, obj)


@app.get("/admin/api/faq", tags=["admin"])
def admin_get_faq(user: str = Depends(verify_admin)):
    return faq_load()


@app.put("/admin/api/faq", tags=["admin"])
def admin_put_faq(payload: Any = Body(...), user: str = Depends(verify_admin)):
    faq_save(payload)
    load_faq()
    build_faq_index()
    return {"ok": True}


@app.post("/admin/api/faq/item", tags=["admin"])
def admin_add_faq_item(item: Any = Body(...), user: str = Depends(verify_admin)):
    data = faq_load()
    if isinstance(data, list):
        data.append(item)
        faq_save(data)
    elif isinstance(data, dict):
        _id = (item or {}).get("id")
        if not _id:
            raise HTTPException(status_code=400, detail="FAQ が dict の場合、item.id が必要です")
        data[str(_id)] = item
        faq_save(data)
    else:
        raise HTTPException(status_code=400, detail="FAQ の形式が不正です")

    load_faq()
    build_faq_index()
    return {"ok": True}


@app.patch("/admin/api/faq/item/{key}", tags=["admin"])
def admin_update_faq_item(key: str, item: Any = Body(...), user: str = Depends(verify_admin)):
    data = faq_load()
    if isinstance(data, list):
        try:
            idx = int(key)
        except:
            raise HTTPException(status_code=400, detail="FAQ が list の場合 key は index（数値）です")
        if idx < 0 or idx >= len(data):
            raise HTTPException(status_code=404, detail="対象が見つかりません")
        data[idx] = item
        faq_save(data)
    elif isinstance(data, dict):
        data[key] = item
        faq_save(data)
    else:
        raise HTTPException(status_code=400, detail="FAQ の形式が不正です")

    load_faq()
    build_faq_index()
    return {"ok": True}


@app.delete("/admin/api/faq/item/{key}", tags=["admin"])
def admin_delete_faq_item(key: str, user: str = Depends(verify_admin)):
    data = faq_load()
    if isinstance(data, list):
        idx = int(key)
        if idx < 0 or idx >= len(data):
            raise HTTPException(status_code=404, detail="対象が見つかりません")
        data.pop(idx)
        faq_save(data)
    elif isinstance(data, dict):
        if key in data:
            del data[key]
            faq_save(data)
    else:
        raise HTTPException(status_code=400, detail="FAQ の形式が不正です")

    load_faq()
    build_faq_index()
    return {"ok": True}


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
# 画面配信（既存フロント + 管理UI）
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

@app.get("/faq/debug", tags=["faq"])
def faq_debug():
    return {
        "faq_file_exists": FAQ_PATH.exists(),
        "faq_items_count": len(faq_items_flat),
        "faq_index_ready": faq_index is not None,
        "sample": faq_items_flat[0] if faq_items_flat else None,
    }
