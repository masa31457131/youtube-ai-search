"""
最適化されたFastAPI サポート検索システム (Render.com対応版)
- 起動時間削減 (lazy loading)
- FAISS インデックス最適化
- 既存の静的ファイル構成との互換性維持
"""

from fastapi import FastAPI, Query, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from contextlib import asynccontextmanager
from functools import lru_cache

from sentence_transformers import SentenceTransformer
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
import unicodedata

# ============================================
# 設定
# ============================================

APP_TITLE = "サポート検索（動画 + FAQ）最適化版"
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)

DEFAULT_TOP_K = 10
DEFAULT_PAGE_LIMIT = 10
MAX_PAGE_LIMIT = 50

# 管理者認証
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "abc123")

BASE_DIR = pathlib.Path(__file__).parent
DATA_PATH = BASE_DIR / "data.json"
SYNONYMS_PATH = BASE_DIR / "synonyms.json"
FAQ_PATH = BASE_DIR / "faq_chatbot_fixed_only.json"
SEARCH_LOG_PATH = BASE_DIR / "search_logs.csv"

# 静的ファイルパス
frontend_path = BASE_DIR / "frontend"
admin_path = BASE_DIR / "admin_ui"

# ============================================
# グローバル状態管理
# ============================================

class AppState:
    """アプリケーション状態の一元管理"""
    def __init__(self):
        # 動画検索用
        self.videos: List[Dict[str, Any]] = []
        self.text_corpus: List[str] = []
        self.synonyms: Dict[str, List[str]] = {}
        self.model: Optional[SentenceTransformer] = None
        self.video_index: Optional[faiss.Index] = None
        self.video_embeddings: Optional[np.ndarray] = None
        
        # FAQ検索用
        self.faq_data: Dict[str, Any] = {}
        self.faq_items_flat: List[Dict[str, Any]] = []
        self.faq_corpus: List[str] = []
        self.faq_index: Optional[faiss.Index] = None
        self.faq_embeddings: Optional[np.ndarray] = None
        
        # 初期化状態
        self.video_loaded = False
        self.faq_loaded = False
        self.model_loaded = False
        
    async def ensure_model_loaded(self):
        """モデルの遅延ロード"""
        if not self.model_loaded:
            print(f"🔄 Loading model: {EMBEDDING_MODEL}")
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            self.model_loaded = True
            print("✅ Model loaded")
    
    async def ensure_video_loaded(self):
        """動画データの遅延ロード"""
        if not self.video_loaded:
            await self.ensure_model_loaded()
            print("🔄 Loading video data...")
            
            # データ読み込み
            if DATA_PATH.exists():
                with open(DATA_PATH, "r", encoding="utf-8") as f:
                    self.videos = json.load(f)
            
            if SYNONYMS_PATH.exists():
                with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
                    self.synonyms = json.load(f)
            
            # コーパス構築
            self.text_corpus = []
            for v in self.videos:
                text = f"{v.get('title', '')} {v.get('description', '')} {v.get('transcript', '')}"
                self.text_corpus.append(normalize_text(text))
            
            # FAISS インデックス構築
            if self.text_corpus:
                self.video_embeddings = self.model.encode(
                    self.text_corpus, 
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                faiss.normalize_L2(self.video_embeddings)
                self.video_index = build_optimized_index(self.video_embeddings)
            
            self.video_loaded = True
            print(f"✅ Video data loaded: {len(self.videos)} videos")
    
    async def ensure_faq_loaded(self):
        """FAQデータの遅延ロード"""
        if not self.faq_loaded:
            await self.ensure_model_loaded()
            print("🔄 Loading FAQ data...")
            
            if FAQ_PATH.exists():
                with open(FAQ_PATH, "r", encoding="utf-8") as f:
                    self.faq_data = json.load(f)
            
            # FAQ アイテムをフラット化
            self.faq_items_flat = []
            for category_key, items in self.faq_data.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            item["category"] = category_key
                            self.faq_items_flat.append(item)
            
            # コーパス構築
            self.faq_corpus = []
            for item in self.faq_items_flat:
                text_parts = [
                    item.get("intent", ""),
                    item.get("question", ""),
                    " ".join(item.get("utterances", [])),
                    " ".join(item.get("keywords", []))
                ]
                combined = " ".join(text_parts)
                self.faq_corpus.append(normalize_text(combined))
            
            # FAISS インデックス構築
            if self.faq_corpus:
                self.faq_embeddings = self.model.encode(
                    self.faq_corpus,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                faiss.normalize_L2(self.faq_embeddings)
                self.faq_index = build_optimized_index(self.faq_embeddings)
            
            self.faq_loaded = True
            print(f"✅ FAQ data loaded: {len(self.faq_items_flat)} items")

state = AppState()

# ============================================
# ユーティリティ関数
# ============================================

@lru_cache(maxsize=1000)
def normalize_text(text: str) -> str:
    """テキスト正規化（キャッシュ付き）"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def expand_with_synonyms(query: str, synonyms: Dict[str, List[str]]) -> str:
    """同義語展開（シンプル版）"""
    expanded_terms = [query]
    for term, syns in synonyms.items():
        if term.lower() in query.lower():
            expanded_terms.extend(syns)
    return " ".join(expanded_terms)

def build_optimized_index(embeddings: np.ndarray) -> faiss.Index:
    """最適化されたFAISSインデックス構築"""
    n, dim = embeddings.shape
    
    if n <= 1000:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    else:
        nlist = min(100, n // 10)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = 10
    
    return index

def log_search(query: str):
    """検索ログ記録"""
    try:
        with open(SEARCH_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(timezone.utc).isoformat(), query])
    except Exception as e:
        print(f"⚠️ Log write failed: {e}")

def parse_logs() -> List[Dict[str, Any]]:
    """ログパース（管理画面用）"""
    if not SEARCH_LOG_PATH.exists():
        return []
    
    rows = []
    with open(SEARCH_LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                try:
                    dt = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
                    rows.append({"dt": dt, "query": row[1]})
                except:
                    pass
    return rows

# ============================================
# 認証
# ============================================

security = HTTPBasic()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """管理者認証"""
    correct_username = secrets.compare_digest(credentials.username, ADMIN_USER)
    correct_password = secrets.compare_digest(credentials.password, ADMIN_PASS)
    
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# ============================================
# Lifespan管理
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """起動時は最小限の初期化のみ"""
    print("🚀 Application starting...")
    
    # ログファイル初期化
    if not SEARCH_LOG_PATH.exists():
        with open(SEARCH_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "query"])
    
    # ディレクトリ作成
    admin_path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Application ready (lazy loading enabled)")
    yield
    print("🛑 Application shutting down...")

# ============================================
# FastAPI アプリケーション
# ============================================

app = FastAPI(title=APP_TITLE, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# ヘルスチェック
# ============================================

@app.get("/health")
async def health_check():
    """ヘルスチェック（Render.com用）"""
    return {
        "status": "healthy",
        "model_loaded": state.model_loaded,
        "video_loaded": state.video_loaded,
        "faq_loaded": state.faq_loaded
    }

# ============================================
# 動画検索エンドポイント
# ============================================

@app.get("/search")
async def search_videos(
    query: str = Query(..., min_length=1),
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    paged: int = Query(0)
):
    """動画検索（ページング対応）"""
    await state.ensure_video_loaded()
    
    if not state.video_index:
        return {"items": [], "has_more": False, "total_visible": 0}
    
    log_search(f"video:{query}")
    
    normalized_query = normalize_text(query)
    expanded_query = expand_with_synonyms(normalized_query, state.synonyms)
    
    query_embedding = state.model.encode([expanded_query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    k = min(offset + limit + 50, len(state.videos))
    distances, indices = state.video_index.search(query_embedding, k)
    
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if 0 <= idx < len(state.videos):
            video = state.videos[idx].copy()
            video["score"] = float(score)
            results.append(video)
    
    total = len(results)
    items = results[offset:offset + limit]
    has_more = (offset + limit) < total
    
    if paged:
        return {
            "items": items,
            "has_more": has_more,
            "total_visible": total,
            "offset": offset,
            "limit": limit
        }
    
    return {"items": items}

# ============================================
# FAQ検索エンドポイント
# ============================================

@app.get("/faq/search")
async def search_faq(
    query: str = Query(..., min_length=1),
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    paged: int = Query(0)
):
    """FAQ検索（ページング対応）"""
    await state.ensure_faq_loaded()
    
    if not state.faq_index:
        return {"items": [], "has_more": False, "total_visible": 0}
    
    log_search(f"faq:{query}")
    
    normalized_query = normalize_text(query)
    query_embedding = state.model.encode([normalized_query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    k = min(offset + limit + 50, len(state.faq_items_flat))
    distances, indices = state.faq_index.search(query_embedding, k)
    
    results = []
    for idx, score in zip(indices[0], distances[0]):
        if 0 <= idx < len(state.faq_items_flat):
            item = state.faq_items_flat[idx].copy()
            item["score"] = float(score)
            results.append(item)
    
    total = len(results)
    items = results[offset:offset + limit]
    has_more = (offset + limit) < total
    
    if paged:
        return {
            "items": items,
            "has_more": has_more,
            "total_visible": total,
            "offset": offset,
            "limit": limit
        }
    
    return {"items": items}

# ============================================
# 管理API - データ編集
# ============================================

@app.get("/admin/api/synonyms", dependencies=[Depends(verify_admin)])
async def get_synonyms():
    """同義語辞書取得"""
    if SYNONYMS_PATH.exists():
        with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@app.put("/admin/api/synonyms", dependencies=[Depends(verify_admin)])
async def update_synonyms(data: dict, background_tasks: BackgroundTasks):
    """同義語辞書更新"""
    with open(SYNONYMS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    return {"status": "ok", "count": len(data)}

async def reload_video_data():
    """動画データのリロード"""
    state.video_loaded = False
    await state.ensure_video_loaded()

@app.post("/admin/api/synonyms/generate", dependencies=[Depends(verify_admin)])
async def generate_synonyms(background_tasks: BackgroundTasks):
    """data.jsonから同義語を生成"""
    await state.ensure_video_loaded()
    
    synonym_map = {}
    for v in state.videos:
        title = v.get("title", "")
        desc = v.get("description", "")
        
        # タイトルから主要キーワード抽出（簡易版）
        words = re.findall(r'[\w]+', title + " " + desc)
        for word in words:
            if len(word) > 2:
                if word not in synonym_map:
                    synonym_map[word] = []
    
    with open(SYNONYMS_PATH, "w", encoding="utf-8") as f:
        json.dump(synonym_map, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    return {"status": "ok", "count": len(synonym_map)}

@app.patch("/admin/api/synonyms/{term}", dependencies=[Depends(verify_admin)])
async def update_synonym_term(term: str, values: List[str], background_tasks: BackgroundTasks):
    """同義語の個別更新"""
    synonyms = {}
    if SYNONYMS_PATH.exists():
        with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
            synonyms = json.load(f)
    
    synonyms[term] = values
    
    with open(SYNONYMS_PATH, "w", encoding="utf-8") as f:
        json.dump(synonyms, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    return {"status": "ok", "term": term}

@app.delete("/admin/api/synonyms/{term}", dependencies=[Depends(verify_admin)])
async def delete_synonym_term(term: str, background_tasks: BackgroundTasks):
    """同義語の個別削除"""
    synonyms = {}
    if SYNONYMS_PATH.exists():
        with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
            synonyms = json.load(f)
    
    if term in synonyms:
        del synonyms[term]
    
    with open(SYNONYMS_PATH, "w", encoding="utf-8") as f:
        json.dump(synonyms, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    return {"status": "ok", "term": term}

@app.get("/admin/api/faq", dependencies=[Depends(verify_admin)])
async def get_faq():
    """FAQ全体取得"""
    if FAQ_PATH.exists():
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@app.put("/admin/api/faq", dependencies=[Depends(verify_admin)])
async def update_faq(data: dict, background_tasks: BackgroundTasks):
    """FAQ全体更新"""
    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_faq_data)
    return {"status": "ok"}

async def reload_faq_data():
    """FAQデータのリロード"""
    state.faq_loaded = False
    await state.ensure_faq_loaded()

# ============================================
# 管理API - FAQ個別編集
# ============================================

@app.get("/admin/api/faq/items", dependencies=[Depends(verify_admin)])
async def list_faq_items(offset: int = 0, limit: int = 50, q: str = ""):
    """FAQ一覧取得（検索・ページング対応）"""
    await state.ensure_faq_loaded()
    
    items = state.faq_items_flat
    
    if q:
        q_lower = q.lower()
        items = [
            item for item in items
            if q_lower in item.get("question", "").lower()
            or q_lower in item.get("category", "").lower()
            or any(q_lower in kw.lower() for kw in item.get("keywords", []))
        ]
    
    total = len(items)
    page_items = items[offset:offset + limit]
    
    return {"items": page_items, "has_more": (offset + limit) < total, "total": total}

@app.post("/admin/api/faq/item", dependencies=[Depends(verify_admin)])
async def create_faq_item(item: dict, background_tasks: BackgroundTasks):
    """FAQ新規作成"""
    faq_id = item.get("id")
    if not faq_id:
        raise HTTPException(400, "ID is required")
    
    await state.ensure_faq_loaded()
    if any(f.get("id") == faq_id for f in state.faq_items_flat):
        raise HTTPException(400, f"ID '{faq_id}' already exists")
    
    category = item.get("category", "その他")
    if category not in state.faq_data:
        state.faq_data[category] = []
    
    state.faq_data[category].append(item)
    
    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        json.dump(state.faq_data, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_faq_data)
    return {"status": "created", "id": faq_id}

@app.patch("/admin/api/faq/item/{item_id}", dependencies=[Depends(verify_admin)])
async def update_faq_item(item_id: str, item: dict, background_tasks: BackgroundTasks):
    """FAQ更新"""
    await state.ensure_faq_loaded()
    
    found = False
    for category, items in state.faq_data.items():
        if isinstance(items, list):
            for i, existing in enumerate(items):
                if existing.get("id") == item_id:
                    state.faq_data[category][i] = item
                    found = True
                    break
        if found:
            break
    
    if not found:
        raise HTTPException(404, f"FAQ item '{item_id}' not found")
    
    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        json.dump(state.faq_data, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_faq_data)
    return {"status": "updated", "id": item_id}

@app.delete("/admin/api/faq/item/{item_id}", dependencies=[Depends(verify_admin)])
async def delete_faq_item(item_id: str, background_tasks: BackgroundTasks):
    """FAQ削除"""
    await state.ensure_faq_loaded()
    
    found = False
    for category, items in state.faq_data.items():
        if isinstance(items, list):
            for i, existing in enumerate(items):
                if existing.get("id") == item_id:
                    del state.faq_data[category][i]
                    found = True
                    break
        if found:
            break
    
    if not found:
        raise HTTPException(404, f"FAQ item '{item_id}' not found")
    
    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        json.dump(state.faq_data, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_faq_data)
    return {"status": "deleted", "id": item_id}

# ============================================
# 管理API - 動画データ
# ============================================

@app.get("/admin/api/videos", dependencies=[Depends(verify_admin)])
async def get_videos():
    """動画データ一覧取得"""
    if not DATA_PATH.exists():
        return []
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        videos = json.load(f)
    
    return videos

@app.post("/admin/api/videos", dependencies=[Depends(verify_admin)])
async def create_video(video_data: dict, background_tasks: BackgroundTasks):
    """動画データ作成"""
    videos = []
    if DATA_PATH.exists():
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            videos = json.load(f)
    
    videos.append(video_data)
    
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(videos, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    return {"status": "created", "video_id": video_data.get("video_id")}

@app.patch("/admin/api/videos/{video_id}", dependencies=[Depends(verify_admin)])
async def update_video(video_id: str, video_data: dict, background_tasks: BackgroundTasks):
    """動画データ更新"""
    if not DATA_PATH.exists():
        raise HTTPException(404, "Data file not found")
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        videos = json.load(f)
    
    found = False
    for i, video in enumerate(videos):
        if video.get("video_id") == video_id:
            videos[i] = video_data
            found = True
            break
    
    if not found:
        raise HTTPException(404, f"Video '{video_id}' not found")
    
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(videos, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    return {"status": "updated", "video_id": video_id}

@app.delete("/admin/api/videos/{video_id}", dependencies=[Depends(verify_admin)])
async def delete_video(video_id: str, background_tasks: BackgroundTasks):
    """動画データ削除"""
    if not DATA_PATH.exists():
        raise HTTPException(404, "Data file not found")
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        videos = json.load(f)
    
    found = False
    for i, video in enumerate(videos):
        if video.get("video_id") == video_id:
            del videos[i]
            found = True
            break
    
    if not found:
        raise HTTPException(404, f"Video '{video_id}' not found")
    
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(videos, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    return {"status": "deleted", "video_id": video_id}

@app.post("/admin/api/videos/import", dependencies=[Depends(verify_admin)])
async def import_videos(import_data: dict, background_tasks: BackgroundTasks):
    """動画データインポート"""
    mode = import_data.get("mode", "merge")
    new_data = import_data.get("data", [])
    
    if not isinstance(new_data, list):
        raise HTTPException(400, "Invalid data format")
    
    added = 0
    updated = 0
    
    if mode == "replace":
        # 全体置換
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        added = len(new_data)
    else:
        # 差分マージ
        existing_videos = []
        if DATA_PATH.exists():
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                existing_videos = json.load(f)
        
        existing_ids = {v.get("video_id"): i for i, v in enumerate(existing_videos)}
        
        for new_video in new_data:
            video_id = new_video.get("video_id")
            if video_id in existing_ids:
                existing_videos[existing_ids[video_id]] = new_video
                updated += 1
            else:
                existing_videos.append(new_video)
                added += 1
        
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(existing_videos, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    return {"status": "imported", "added": added, "updated": updated}

async def reload_video_data():
    """動画データ再読み込み"""
    state.video_loaded = False
    # 次回検索時に自動的に再ロード

# ============================================
# 管理API - YouTube文字起こし
# ============================================

@app.post("/admin/api/youtube/fetch", dependencies=[Depends(verify_admin)])
async def fetch_youtube_videos(request_data: dict):
    """YouTubeチャンネルから動画リストを取得"""
    try:
        from googleapiclient.discovery import build
        
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise HTTPException(400, "YOUTUBE_API_KEY environment variable not set")
        
        channel_url = request_data.get("channel_url", "")
        max_results = request_data.get("max_results", 50)
        
        # チャンネルIDを抽出
        channel_id = None
        if "/c/" in channel_url or "/channel/" in channel_url or "/@" in channel_url:
            # チャンネル名からIDを取得する必要がある
            # 簡略化のため、ユーザーにチャンネルIDを直接入力してもらう方式も検討
            parts = channel_url.rstrip('/').split('/')
            channel_name = parts[-1]
            
            youtube = build('youtube', 'v3', developerKey=api_key)
            
            # チャンネル名から検索
            search_response = youtube.search().list(
                q=channel_name,
                type='channel',
                part='id',
                maxResults=1
            ).execute()
            
            if search_response.get('items'):
                channel_id = search_response['items'][0]['id']['channelId']
        else:
            raise HTTPException(400, "Invalid channel URL format")
        
        if not channel_id:
            raise HTTPException(404, "Channel not found")
        
        # チャンネルのアップロードプレイリストIDを取得
        youtube = build('youtube', 'v3', developerKey=api_key)
        channel_response = youtube.channels().list(
            id=channel_id,
            part='contentDetails'
        ).execute()
        
        if not channel_response.get('items'):
            raise HTTPException(404, "Channel not found")
        
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # プレイリストから動画を取得
        videos = []
        next_page_token = None
        
        while len(videos) < max_results:
            playlist_response = youtube.playlistItems().list(
                playlistId=uploads_playlist_id,
                part='snippet',
                maxResults=min(50, max_results - len(videos)),
                pageToken=next_page_token
            ).execute()
            
            for item in playlist_response.get('items', []):
                snippet = item['snippet']
                video_id = snippet['resourceId']['videoId']
                
                videos.append({
                    'video_id': video_id,
                    'title': snippet['title'],
                    'description': snippet['description'],
                    'thumbnail': snippet['thumbnails'].get('high', {}).get('url', ''),
                    'url': f'https://www.youtube.com/watch?v={video_id}',
                    'published_at': snippet['publishedAt']
                })
            
            next_page_token = playlist_response.get('nextPageToken')
            if not next_page_token:
                break
        
        return {
            'status': 'success',
            'channel_id': channel_id,
            'videos': videos,
            'total': len(videos)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to fetch YouTube videos: {str(e)}")

@app.post("/admin/api/youtube/transcribe", dependencies=[Depends(verify_admin)])
async def transcribe_youtube_video(request_data: dict, background_tasks: BackgroundTasks):
    """YouTube動画を文字起こし"""
    try:
        video_id = request_data.get("video_id")
        if not video_id:
            raise HTTPException(400, "video_id is required")
        
        # 既存のdata.jsonを読み込み
        existing_videos = []
        if DATA_PATH.exists():
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                existing_videos = json.load(f)
        
        # 既に存在するかチェック
        for video in existing_videos:
            if video.get('video_id') == video_id:
                return {
                    'status': 'already_exists',
                    'message': 'Video already transcribed',
                    'video_id': video_id
                }
        
        # バックグラウンドで文字起こし処理
        background_tasks.add_task(process_transcription, video_id, request_data)
        
        return {
            'status': 'processing',
            'message': 'Transcription started in background',
            'video_id': video_id
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to start transcription: {str(e)}")

@app.post("/admin/api/youtube/sync", dependencies=[Depends(verify_admin)])
async def sync_with_youtube(request_data: dict):
    """YouTubeチャンネルとdata.jsonを同期（差分検出）"""
    try:
        from googleapiclient.discovery import build
        
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise HTTPException(400, "YOUTUBE_API_KEY environment variable not set")
        
        channel_url = request_data.get("channel_url", "")
        
        # チャンネルIDを抽出
        channel_id = None
        if "/c/" in channel_url or "/channel/" in channel_url or "/@" in channel_url:
            parts = channel_url.rstrip('/').split('/')
            channel_name = parts[-1]
            
            youtube = build('youtube', 'v3', developerKey=api_key)
            
            search_response = youtube.search().list(
                q=channel_name,
                type='channel',
                part='id',
                maxResults=1
            ).execute()
            
            if search_response.get('items'):
                channel_id = search_response['items'][0]['id']['channelId']
        
        if not channel_id:
            raise HTTPException(404, "Channel not found")
        
        # チャンネルの全動画を取得
        youtube = build('youtube', 'v3', developerKey=api_key)
        channel_response = youtube.channels().list(
            id=channel_id,
            part='contentDetails'
        ).execute()
        
        if not channel_response.get('items'):
            raise HTTPException(404, "Channel not found")
        
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # プレイリストから全動画を取得
        youtube_videos = []
        next_page_token = None
        
        while True:
            playlist_response = youtube.playlistItems().list(
                playlistId=uploads_playlist_id,
                part='snippet',
                maxResults=50,
                pageToken=next_page_token
            ).execute()
            
            for item in playlist_response.get('items', []):
                snippet = item['snippet']
                video_id = snippet['resourceId']['videoId']
                
                youtube_videos.append({
                    'video_id': video_id,
                    'title': snippet['title'],
                    'description': snippet['description'],
                    'thumbnail': snippet['thumbnails'].get('high', {}).get('url', ''),
                    'url': f'https://www.youtube.com/watch?v={video_id}',
                    'published_at': snippet['publishedAt']
                })
            
            next_page_token = playlist_response.get('nextPageToken')
            if not next_page_token:
                break
        
        # 既存のdata.jsonを読み込み
        existing_videos = []
        if DATA_PATH.exists():
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                existing_videos = json.load(f)
        
        # 差分を計算
        youtube_ids = set(v['video_id'] for v in youtube_videos)
        existing_ids = set(v.get('video_id') for v in existing_videos)
        
        # YouTubeにあるが、data.jsonにない（追加すべき動画）
        missing_in_data = [v for v in youtube_videos if v['video_id'] not in existing_ids]
        
        # data.jsonにあるが、YouTubeにない（削除すべき動画）
        missing_in_youtube = [v for v in existing_videos if v.get('video_id') not in youtube_ids]
        
        return {
            'status': 'success',
            'total_youtube': len(youtube_videos),
            'total_data': len(existing_videos),
            'missing_in_data': missing_in_data,
            'missing_in_youtube': missing_in_youtube,
            'youtube_ids': list(youtube_ids),
            'existing_ids': list(existing_ids)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to sync with YouTube: {str(e)}")

@app.post("/admin/api/youtube/cleanup", dependencies=[Depends(verify_admin)])
async def cleanup_orphaned_videos(request_data: dict, background_tasks: BackgroundTasks):
    """YouTubeに存在しない動画をdata.jsonから削除"""
    try:
        video_ids_to_delete = request_data.get("video_ids", [])
        
        if not video_ids_to_delete:
            raise HTTPException(400, "video_ids is required")
        
        if not DATA_PATH.exists():
            raise HTTPException(404, "Data file not found")
        
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            videos = json.load(f)
        
        # 削除対象以外の動画を残す
        filtered_videos = [v for v in videos if v.get('video_id') not in video_ids_to_delete]
        
        # no番号を振り直し
        for i, video in enumerate(filtered_videos, 1):
            video['no'] = i
        
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(filtered_videos, f, ensure_ascii=False, indent=2)
        
        background_tasks.add_task(reload_video_data)
        
        return {
            'status': 'success',
            'deleted_count': len(video_ids_to_delete),
            'remaining_count': len(filtered_videos)
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to cleanup: {str(e)}")

async def process_transcription(video_id: str, video_data: dict):
    """文字起こし処理（バックグラウンド）"""
    import tempfile
    import subprocess
    
    try:
        # yt-dlpで音声ダウンロード
        audio_path = tempfile.mktemp(suffix='.mp3')
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        
        subprocess.run([
            'yt-dlp',
            '-x',
            '--audio-format', 'mp3',
            '-o', audio_path,
            video_url
        ], check=True, capture_output=True)
        
        # Whisperで文字起こし（簡易版）
        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, language='ja')
        transcript = result['text']
        
        # data.jsonに追加
        existing_videos = []
        if DATA_PATH.exists():
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                existing_videos = json.load(f)
        
        # 新しいno番号を生成
        max_no = max([v.get('no', 0) for v in existing_videos], default=0)
        
        new_video = {
            'no': max_no + 1,
            'video_id': video_id,
            'title': video_data.get('title', ''),
            'description': video_data.get('description', ''),
            'transcript': transcript,
            'url': video_data.get('url', ''),
            'thumbnail': video_data.get('thumbnail', ''),
            'status': 'completed'
        }
        
        existing_videos.append(new_video)
        
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(existing_videos, f, ensure_ascii=False, indent=2)
        
        # 一時ファイル削除
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # 動画データ再読み込み
        await reload_video_data()
        
    except Exception as e:
        print(f"Transcription error for {video_id}: {str(e)}")
        # エラー時もdata.jsonに記録（status: failed）
        existing_videos = []
        if DATA_PATH.exists():
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                existing_videos = json.load(f)
        
        max_no = max([v.get('no', 0) for v in existing_videos], default=0)
        
        error_video = {
            'no': max_no + 1,
            'video_id': video_id,
            'title': video_data.get('title', ''),
            'description': video_data.get('description', ''),
            'transcript': f'Error: {str(e)}',
            'url': video_data.get('url', ''),
            'thumbnail': video_data.get('thumbnail', ''),
            'status': 'failed'
        }
        
        existing_videos.append(error_video)
        
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(existing_videos, f, ensure_ascii=False, indent=2)

# ============================================
# 管理API - ログ
# ============================================

@app.get("/admin/api/logs/months", dependencies=[Depends(verify_admin)])
async def get_log_months():
    """利用可能な月一覧"""
    rows = parse_logs()
    months = set(r["dt"].strftime("%Y-%m") for r in rows)
    return {"months": sorted(months, reverse=True)}

@app.get("/admin/api/logs/summary", dependencies=[Depends(verify_admin)])
async def get_log_summary(month: str = Query(...)):
    """月別サマリー"""
    rows = parse_logs()
    day_counter = Counter()
    query_counter = Counter()
    
    for r in rows:
        if r["dt"].strftime("%Y-%m") == month:
            day_counter[r["dt"].strftime("%Y-%m-%d")] += 1
            query_counter[r["query"]] += 1
    
    days = [{"day": d, "count": c} for d, c in sorted(day_counter.items())]
    top_queries = [{"query": q, "count": c} for q, c in query_counter.most_common(50)]
    
    return {"month": month, "days": days, "top_queries": top_queries}

@app.get("/admin/api/logs/export", dependencies=[Depends(verify_admin)])
async def export_logs():
    """ログCSVエクスポート"""
    if not SEARCH_LOG_PATH.exists():
        raise HTTPException(404, "No logs found")
    
    csv_data = SEARCH_LOG_PATH.read_text(encoding="utf-8")
    headers = {"Content-Disposition": 'attachment; filename="search_logs.csv"'}
    return StreamingResponse(iter([csv_data]), media_type="text/csv", headers=headers)

# ============================================
# 静的ファイル配信
# ============================================

# フロントエンド静的ファイル
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# 管理画面静的ファイル
app.mount("/admin/static", StaticFiles(directory=admin_path), name="admin_static")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_index():
    """検索画面"""
    index_file = frontend_path / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return index_file.read_text(encoding="utf-8")

@app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_home():
    """管理画面トップ"""
    f = admin_path / "index.html"
    if not f.exists():
        return HTMLResponse("<h1>admin index.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/dashboard", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_dashboard():
    """管理画面 - ダッシュボード"""
    f = admin_path / "dashboard.html"
    if not f.exists():
        return HTMLResponse("<h1>admin dashboard.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/videos", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_videos():
    """管理画面 - 動画データ"""
    f = admin_path / "videos.html"
    if not f.exists():
        return HTMLResponse("<h1>admin videos.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/synonyms", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_synonyms():
    """管理画面 - Synonyms"""
    f = admin_path / "synonyms.html"
    if not f.exists():
        return HTMLResponse("<h1>admin synonyms.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/faq", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_faq():
    """管理画面 - FAQ"""
    f = admin_path / "faq.html"
    if not f.exists():
        return HTMLResponse("<h1>admin faq.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/logs", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_logs():
    """管理画面 - ログ"""
    f = admin_path / "logs.html"
    if not f.exists():
        return HTMLResponse("<h1>admin logs.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

# .html拡張子付きのルートも追加
@app.get("/admin/dashboard.html", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_dashboard_html():
    """管理画面 - ダッシュボード (.html)"""
    f = admin_path / "dashboard.html"
    if not f.exists():
        return HTMLResponse("<h1>admin dashboard.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/videos.html", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_videos_html():
    """管理画面 - 動画データ (.html)"""
    f = admin_path / "videos.html"
    if not f.exists():
        return HTMLResponse("<h1>admin videos.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/synonyms.html", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_synonyms_html():
    """管理画面 - Synonyms (.html)"""
    f = admin_path / "synonyms.html"
    if not f.exists():
        return HTMLResponse("<h1>admin synonyms.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/faq.html", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_faq_html():
    """管理画面 - FAQ (.html)"""
    f = admin_path / "faq.html"
    if not f.exists():
        return HTMLResponse("<h1>admin faq.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/logs.html", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_logs_html():
    """管理画面 - ログ (.html)"""
    f = admin_path / "logs.html"
    if not f.exists():
        return HTMLResponse("<h1>admin logs.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
