from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from sentence_transformers import SentenceTransformer
import torch
import faiss
import json
import os
import pathlib
import io
import csv
import secrets
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np

# ============================================
# 設定
# ============================================

APP_TITLE = "音声検索AI - サポート検索 (動画 + FAQ統合)"

# 埋め込みモデル（日本語を含む多言語に強い & 高速）
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)

# 検索結果の件数
DEFAULT_TOP_K = 10
DEFAULT_FAQ_TOP_K = 5

# 管理用のBasic認証
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "changeme")  # 本番では必ず変更してください

# ファイルパス（data.json / synonyms.json / faq_chatbot_fixed_only.json / search_logs.csv を同じディレクトリに置く）
BASE_DIR = pathlib.Path(__file__).parent
DATA_PATH = BASE_DIR / "data.json"
SYNONYMS_PATH = BASE_DIR / "synonyms.json"

FAQ_PATH = BASE_DIR / "faq_chatbot_fixed_only.json"  # FIX済みFAQのみ（誘導系は除外）
SEARCH_LOG_PATH = BASE_DIR / "search_logs.csv"


# ============================================
# グローバル変数
# ============================================

app = FastAPI(title=APP_TITLE)

# CORS（フロントが別ドメインでも動くようにゆるめに許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBasic()

videos: List[Dict[str, Any]] = []
text_corpus: List[str] = []
synonyms: Dict[str, List[str]] = {}

faqs: List[Dict[str, Any]] = []
faq_corpus: List[str] = []

model: Optional[SentenceTransformer] = None
index: Optional[faiss.IndexFlatIP] = None       # 動画用（コサイン類似度：内積Index）
faq_index: Optional[faiss.IndexFlatIP] = None   # FAQ用（コサイン類似度：内積Index）


# ============================================
# 前処理ユーティリティ
# ============================================

def normalize_text(text: str) -> str:
    """
    簡易正規化:
    - 英数字を小文字化
    - 連続スペースを 1つに
    - 前後の空白を除去
    """
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data() -> None:
    """
    動画データ (data.json) と 類義語辞書 (synonyms.json) を読み込む。
    ついでに検索用コーパスも構築する。
    """
    global videos, synonyms, text_corpus

    if not DATA_PATH.exists():
        raise RuntimeError(f"data.json が見つかりません: {DATA_PATH}")
    with open(DATA_PATH, encoding="utf-8") as f:
        videos = json.load(f)

    if SYNONYMS_PATH.exists():
        with open(SYNONYMS_PATH, encoding="utf-8") as f:
            synonyms = json.load(f)
    else:
        synonyms = {}

    # 検索用コーパスを構築
    text_corpus = []
    for v in videos:
        title = normalize_text(v.get("title", "") or "")
        desc = normalize_text(v.get("description", "") or "")
        trans = normalize_text(v.get("transcript", "") or "")

        # タイトルに重みを持たせる（2倍くらい）
        combined = f"{title} [SEP] {desc} [SEP] {trans}"
        combined_weighted = f"{title} {title} {combined}"
        text_corpus.append(combined_weighted)


def load_faq() -> None:
    """
    FIX済みFAQ (faq_chatbot_fixed_only.json) を読み込み、検索コーパスを構築する。
    期待形式: {"meta": {...}, "faqs":[{id, question, utterances?, steps? ...}, ...]}
    """
    global faqs, faq_corpus

    if not FAQ_PATH.exists():
        faqs = []
        faq_corpus = []
        return

    with open(FAQ_PATH, encoding="utf-8") as f:
        payload = json.load(f)

    faqs = payload.get("faqs", []) if isinstance(payload, dict) else []
    faq_corpus = []
    for item in faqs:
        q = normalize_text(item.get("question", "") or "")
        cat = normalize_text(item.get("category", "") or "")
        intent = normalize_text(item.get("intent", "") or "")

        utts = item.get("utterances", []) or []
        utts_norm = [normalize_text(u) for u in utts if isinstance(u, str)]

        # 質問 + 表記ゆれ(utterances) + カテゴリ/意図 をまとめて埋め込み対象にする
        joined = " [SEP] ".join([q, " ".join(utts_norm), cat, intent]).strip()
        faq_corpus.append(joined)


def get_model() -> SentenceTransformer:
    """
    SentenceTransformer モデルを lazy にロード
    """
    global model
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return model


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def build_search_index() -> None:
    """
    動画コーパスから FAISS IndexFlatIP を構築（コサイン類似度）
    """
    global index

    if not text_corpus:
        index = None
        return

    m = get_model()
    embeddings = m.encode(text_corpus, convert_to_numpy=True, show_progress_bar=False)
    embeddings = _normalize_embeddings(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内積 = コサイン類似度（正規化済みベクトル）
    index.add(embeddings)


def build_faq_index() -> None:
    """
    FAQコーパスから FAISS IndexFlatIP を構築（コサイン類似度）
    """
    global faq_index

    if not faq_corpus:
        faq_index = None
        return

    m = get_model()
    embeddings = m.encode(faq_corpus, convert_to_numpy=True, show_progress_bar=False)
    embeddings = _normalize_embeddings(embeddings)

    dim = embeddings.shape[1]
    faq_index = faiss.IndexFlatIP(dim)
    faq_index.add(embeddings)


def expand_query_with_synonyms(query: str) -> str:
    """
    クエリを類義語付きで拡張する。
    - スペースでトークン分割して key と一致するものは synonyms[key] を展開
    - クエリ文字列に key が部分一致した場合も類義語を追加（例: 「プロット 出力」など）
    - 重複は順番を保ったまま削除
    """
    if not synonyms:
        return query

    original_query = query
    tokens = list(filter(None, re.split(r"\s+", original_query)))

    expanded: List[str] = []

    # 1) トークン単位での類義語展開
    for t in tokens:
        expanded.append(t)
        if t in synonyms:
            expanded.extend(synonyms[t])

    # 2) 部分一致による展開
    for key, syns in synonyms.items():
        if key in original_query and key not in tokens:
            expanded.append(key)
            expanded.extend(syns)

    # 3) 順番を保ったまま重複を除去
    seen = set()
    unique: List[str] = []
    for w in expanded:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    return " ".join(unique)


def _embed_query(query: str) -> np.ndarray:
    m = get_model()
    normalized_query = normalize_text(query)
    q_for_embed = expand_query_with_synonyms(normalized_query)
    q_emb = m.encode([q_for_embed], convert_to_numpy=True, show_progress_bar=False)
    q_emb = _normalize_embeddings(q_emb)
    return q_emb


def search_videos(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    動画: ベクトル検索 + タイトル/説明の軽いキーワード補正
    """
    if index is None or not videos:
        return []

    normalized_query = normalize_text(query)
    q_emb = _embed_query(normalized_query)

    k = min(top_k * 5, len(videos))
    sims, idxs = index.search(q_emb, k)
    sims = sims[0]
    idxs = idxs[0]

    query_tokens = set(normalized_query.split())
    scored: List[Tuple[float, Dict[str, Any]]] = []

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
        scored.append((final_score, v))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [v for _, v in scored[:top_k]]


def search_faq(query: str, top_k: int = DEFAULT_FAQ_TOP_K) -> List[Dict[str, Any]]:
    """
    FAQ: ベクトル検索（質問 + 表記ゆれを対象）
    返却はチャットで読みやすい steps を含めた構造にする。
    """
    if faq_index is None or not faqs:
        return []

    normalized_query = normalize_text(query)
    q_emb = _embed_query(normalized_query)

    k = min(top_k, len(faqs))
    sims, idxs = faq_index.search(q_emb, k)
    sims = sims[0]
    idxs = idxs[0]

    results: List[Dict[str, Any]] = []
    for sim, idx in zip(sims, idxs):
        if idx < 0 or idx >= len(faqs):
            continue
        item = faqs[idx]

        results.append({
            "id": item.get("id"),
            "category": item.get("category", ""),
            "intent": item.get("intent", ""),
            "question": item.get("question", ""),
            "steps": item.get("steps", []) or [],
            "score": float(sim),
        })
    return results


def log_search_query(query: str, hits_count: int, kind: str = "video") -> None:
    """
    検索ログを CSV に保存
    """
    header = ["timestamp", "kind", "query", "hits"]
    now = datetime.utcnow().isoformat()
    row = [now, kind, query, str(hits_count)]
    file_exists = SEARCH_LOG_PATH.exists()

    with open(SEARCH_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """
    管理者向けBasic認証
    """
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USER)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASS)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# ============================================
# FastAPI イベント & エンドポイント
# ============================================

@app.on_event("startup")
def on_startup() -> None:
    """
    アプリ起動時にデータ読み込みとインデックス構築をまとめて実行
    """
    load_data()
    load_faq()
    build_search_index()
    build_faq_index()


@app.get("/search", summary="動画検索", tags=["search"])
def search_endpoint(
    query: str = Query(..., min_length=1, description="検索クエリ（単語1つでも、短い文章でもOKです）"),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=50, description="返す件数"),
):
    """
    動画検索エンドポイント（従来互換）。
    """
    results = search_videos(query, top_k=top_k)
    log_search_query(query, len(results), kind="video")
    return results


@app.get("/faq/search", summary="FAQ検索（候補提示）", tags=["faq"])
def faq_search_endpoint(
    query: str = Query(..., min_length=1, description="検索クエリ"),
    top_k: int = Query(DEFAULT_FAQ_TOP_K, ge=1, le=20, description="返す件数"),
):
    """
    FAQ候補を返す（UI側で「FAQ候補→選択」表示に利用）。
    """
    results = search_faq(query, top_k=top_k)
    log_search_query(query, len(results), kind="faq")
    return results


@app.get("/faq/{faq_id}", summary="FAQ詳細", tags=["faq"])
def faq_get_endpoint(faq_id: str):
    """
    FAQ ID から詳細を取得
    """
    for item in faqs:
        if item.get("id") == faq_id:
            return {
                "id": item.get("id"),
                "category": item.get("category", ""),
                "intent": item.get("intent", ""),
                "question": item.get("question", ""),
                "steps": item.get("steps", []) or [],
            }
    raise HTTPException(status_code=404, detail="FAQ が見つかりません")


@app.get("/logs/export", summary="検索ログCSVダウンロード", tags=["admin"])
def export_logs(username: str = Depends(verify_admin)):
    """
    検索ログを CSV としてダウンロード
    """
    if not SEARCH_LOG_PATH.exists():
        raise HTTPException(status_code=404, detail="検索ログがまだありません")

    with open(SEARCH_LOG_PATH, "r", encoding="utf-8") as f:
        csv_data = f.read()

    headers = {"Content-Disposition": 'attachment; filename="search_logs.csv"'}
    return StreamingResponse(iter([csv_data]), media_type="text/csv", headers=headers)


# ============================================
# フロントエンド（静的ファイル）配信
# ============================================

frontend_path = BASE_DIR / "frontend"
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_index():
    index_file = frontend_path / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html が見つかりません")
    return index_file.read_text(encoding="utf-8")
