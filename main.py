from fastapi import FastAPI, Query, HTTPException, Depends
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
import io
import csv
import secrets
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
import numpy as np

# ============================================
# 設定
# ============================================

APP_TITLE = "音声検索AI - サポート検索 (高速・高精度版)"

# 埋め込みモデル（日本語を含む多言語に強い & 高速）
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)

# 検索結果の件数
DEFAULT_TOP_K = 10

# 管理用のBasic認証
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "changeme")  # 本番では必ず変更してください

# ファイルパス
BASE_DIR = pathlib.Path(__file__).parent
DATA_PATH = BASE_DIR / "data.json"
SYNONYMS_PATH = BASE_DIR / "synonyms.json"
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

model: Optional[SentenceTransformer] = None
index: Optional[faiss.IndexFlatIP] = None  # コサイン類似度用に内積Indexを使用


# ============================================
# 前処理ユーティリティ
# ============================================

def normalize_text(text: str) -> str:
    """ひらがな/カタカナはそのまま、英数字は小文字・空白整理だけ行う簡易正規化"""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data() -> None:
    """動画データと類義語辞書を読み込む"""
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
        title = normalize_text(v.get("title", ""))
        desc = normalize_text(v.get("description", ""))
        trans = normalize_text(v.get("transcript", ""))

        # タイトルを重み付け（2倍）して精度を上げる
        combined = f"{title} [SEP] {desc} [SEP] {trans}"
        combined_weighted = f"{title} {title} {combined}"
        text_corpus.append(combined_weighted)


def get_model() -> SentenceTransformer:
    """SentenceTransformer モデルを lazy にロード"""
    global model
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return model


def build_search_index() -> None:
    """テキストコーパスから FAISS IndexFlatIP を構築（コサイン類似度）"""
    global index

    if not text_corpus:
        index = None
        return

    m = get_model()
    embeddings = m.encode(text_corpus, convert_to_numpy=True, show_progress_bar=False)

    # コサイン類似度用に正規化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内積 = コサイン類似度（正規化済みベクトル）
    index.add(embeddings)


def expand_query_with_synonyms(query: str) -> str:
    """クエリに類義語を足して検索の網を広げる"""
    if not synonyms:
        return query

    tokens = list(filter(None, re.split(r"\s+", query)))
    expanded: List[str] = []
    for t in tokens:
        expanded.append(t)
        if t in synonyms:
            expanded.extend(synonyms[t])
    return " ".join(expanded)


def search_core(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """ベクトル検索 + タイトルキーワード補正で精度を高めた検索"""
    if index is None or not videos:
        return []

    m = get_model()

    # 正規化したクエリ
    normalized_query = normalize_text(query)

    # 類義語展開したクエリで埋め込みを作成
    q_for_embed = expand_query_with_synonyms(normalized_query)
    q_emb = m.encode([q_for_embed], convert_to_numpy=True, show_progress_bar=False)

    # 正規化（コサイン類似度用）
    norms = np.linalg.norm(q_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    q_emb = q_emb / norms

    # 一度広めに候補取得（速度と精度のバランスのため top_k * 5）
    k = min(top_k * 5, len(videos))
    sims, idxs = index.search(q_emb, k)
    sims = sims[0]
    idxs = idxs[0]

    query_tokens = set(normalized_query.split())
    results: List[tuple[float, Dict[str, Any]]] = []

    for sim, idx in zip(sims, idxs):
        if idx < 0 or idx >= len(videos):
            continue
        v = videos[idx]

        # タイトル・説明を使った簡易キーワードスコア
        text_for_kw = normalize_text(
            (v.get("title", "") or "") + " " + (v.get("description", "") or "")
        )
        kw_score = 0.0
        if query_tokens:
            hit = sum(1 for t in query_tokens if t and t in text_for_kw)
            kw_score = hit / len(query_tokens)

        # ベクトル類似度とキーワードスコアを統合
        final_score = 0.85 * float(sim) + 0.15 * kw_score

        results.append((final_score, v))

    # スコアでソートして上位 top_k 件を返す
    results.sort(key=lambda x: x[0], reverse=True)
    top = results[:top_k]
    return [v for _, v in top]


def log_search_query(query: str, hits_count: int) -> None:
    """検索ログを CSV に保存"""
    header = ["timestamp", "query", "hits"]
    now = datetime.utcnow().isoformat()
    row = [now, query, str(hits_count)]
    file_exists = SEARCH_LOG_PATH.exists()

    with open(SEARCH_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def verify_admin(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Basic 認証の検証"""
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
    """アプリ起動時にデータ読み込みとインデックス構築をまとめて実行"""
    load_data()
    build_search_index()


@app.get("/search", summary="動画検索", tags=["search"])
def search_endpoint(
    query: str = Query(..., min_length=1, description="検索クエリ"),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=50, description="返す件数"),
):
    """クエリに応じて動画リストを返すエンドポイント"""
    results = search_core(query, top_k=top_k)
    log_search_query(query, len(results))
    return results


@app.get("/logs/export", summary="検索ログCSVダウンロード", tags=["admin"])
def export_logs(username: str = Depends(verify_admin)):
    """検索ログを CSV としてダウンロード"""
    if not SEARCH_LOG_PATH.exists():
        raise HTTPException(status_code=404, detail="検索ログがまだありません")

    with open(SEARCH_LOG_PATH, "r", encoding="utf-8") as f:
        csv_data = f.read()

    headers = {
        "Content-Disposition": 'attachment; filename="search_logs.csv"'
    }
    return StreamingResponse(
        iter([csv_data]),
        media_type="text/csv",
        headers=headers,
    )


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
