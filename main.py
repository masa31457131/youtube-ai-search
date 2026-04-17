"""
æœ€é©åŒ–ã•ã‚ŒãŸFastAPI ã‚µãƒãƒ¼ãƒˆæ¤œç´¢ã‚·ã‚¹ãƒ†ãƒ  (Render.comå¯¾å¿œç‰ˆ)
- èµ·å‹•æ™‚é–“å‰Šæ¸› (lazy loading)
- FAISS ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹æœ€é©åŒ–
- æ—¢å­˜ã®é™çš„ãƒ•ã‚¡ã‚¤ãƒ«æ§‹æˆã¨ã®äº’æ›æ€§ç¶­æŒ
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
# è¨­å®š
# ============================================

APP_TITLE = "ã‚µãƒãƒ¼ãƒˆæ¤œç´¢ï¼ˆå‹•ç”» + FAQï¼‰æœ€é©åŒ–ç‰ˆ"
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)

DEFAULT_TOP_K = 10
DEFAULT_PAGE_LIMIT = 10
MAX_PAGE_LIMIT = 50
DEFAULT_SIMILARITY_THRESHOLD = 0.3  # 類似度スコアのしきい値（0.0-1.0）
SEMANTIC_WEIGHT = 0.6  # セマンティック検索の重み（デフォルト: 60%）
TITLE_WEIGHT = 0.4     # タイトル一致の重み（デフォルト: 40%）

# ç®¡ç†è€…èªè¨¼
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "abc123")

BASE_DIR = pathlib.Path(__file__).parent
DATA_PATH = BASE_DIR / "data.json"
SYNONYMS_PATH = BASE_DIR / "synonyms.json"
FAQ_PATH = BASE_DIR / "faq.json"
CONFIG_PATH = BASE_DIR / "config.json"
USERS_PATH = BASE_DIR / "users.json"
SEARCH_LOG_PATH = BASE_DIR / "search_logs.csv"

# é™çš„ãƒ•ã‚¡ã‚¤ãƒ«ãƒ‘ã‚¹
frontend_path = BASE_DIR / "frontend"
admin_path = BASE_DIR / "admin_ui"

# ============================================
# ã‚°ãƒ­ãƒ¼ãƒãƒ«çŠ¶æ…‹ç®¡ç†
# ============================================

class AppState:
    """ã‚¢ãƒ—ãƒªã‚±ãƒ¼ã‚·ãƒ§ãƒ³çŠ¶æ…‹ã®ä¸€å…ƒç®¡ç†"""
    def __init__(self):
        # å‹•ç”»æ¤œç´¢ç”¨
        self.videos: List[Dict[str, Any]] = []
        self.text_corpus: List[str] = []
        self.synonyms: Dict[str, List[str]] = {}
        self.model: Optional[SentenceTransformer] = None
        self.video_index: Optional[faiss.Index] = None
        self.video_embeddings: Optional[np.ndarray] = None
        
        # FAQæ¤œç´¢ç”¨
        self.faq_data: Dict[str, Any] = {}
        self.faq_items_flat: List[Dict[str, Any]] = []
        self.faq_corpus: List[str] = []
        self.faq_index: Optional[faiss.Index] = None
        self.faq_embeddings: Optional[np.ndarray] = None
        
        # åˆæœŸåŒ–çŠ¶æ…‹
        self.video_loaded = False
        self.faq_loaded = False
        self.model_loaded = False
        
    async def ensure_model_loaded(self):
        """ãƒ¢ãƒ‡ãƒ«ã®é…å»¶ãƒ­ãƒ¼ãƒ‰"""
        if not self.model_loaded:
            print(f"ðŸ”„ Loading model: {EMBEDDING_MODEL}")
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            self.model_loaded = True
            print("âœ… Model loaded")
    
    async def ensure_video_loaded(self):
        """å‹•ç”»ãƒ‡ãƒ¼ã‚¿ã®é…å»¶ãƒ­ãƒ¼ãƒ‰"""
        if not self.video_loaded:
            await self.ensure_model_loaded()
            print("ðŸ”„ Loading video data...")
            
            # ãƒ‡ãƒ¼ã‚¿èª­ã¿è¾¼ã¿
            if DATA_PATH.exists():
                with open(DATA_PATH, "r", encoding="utf-8") as f:
                    self.videos = json.load(f)
            
            if SYNONYMS_PATH.exists():
                with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
                    self.synonyms = json.load(f)
            
            # ã‚³ãƒ¼ãƒ‘ã‚¹æ§‹ç¯‰
            self.text_corpus = []
            for v in self.videos:
                text = f"{v.get('title', '')} {v.get('description', '')} {v.get('transcript', '')}"
                self.text_corpus.append(normalize_text(text))
            
            # FAISS ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹æ§‹ç¯‰
            if self.text_corpus:
                self.video_embeddings = self.model.encode(
                    self.text_corpus, 
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                faiss.normalize_L2(self.video_embeddings)
                self.video_index = build_optimized_index(self.video_embeddings)
            
            self.video_loaded = True
            print(f"âœ… Video data loaded: {len(self.videos)} videos")
    
    async def ensure_faq_loaded(self):
        """FAQãƒ‡ãƒ¼ã‚¿ã®é…å»¶ãƒ­ãƒ¼ãƒ‰"""
        if not self.faq_loaded:
            await self.ensure_model_loaded()
            print("ðŸ”„ Loading FAQ data...")
            
            if FAQ_PATH.exists():
                print(f"✅ FAQ file found: {FAQ_PATH}")
                try:
                    # UTF-8で読み込み試行
                    with open(FAQ_PATH, "r", encoding="utf-8") as f:
                        self.faq_data = json.load(f)
                    print(f"✅ FAQ file loaded (UTF-8), keys: {list(self.faq_data.keys())}")
                except UnicodeDecodeError:
                    # UTF-8で失敗した場合、Shift-JIS (cp932) を試行
                    print(f"⚠️ UTF-8 decode failed, trying cp932...")
                    try:
                        with open(FAQ_PATH, "r", encoding="cp932") as f:
                            self.faq_data = json.load(f)
                        print(f"✅ FAQ file loaded (cp932), keys: {list(self.faq_data.keys())}")
                    except Exception as e:
                        print(f"❌ Failed to load FAQ file with cp932: {e}")
                        self.faq_data = {}
                except Exception as e:
                    print(f"❌ Failed to load FAQ file: {e}")
                    self.faq_data = {}
            else:
                print(f"❌ FAQ file not found: {FAQ_PATH}")
                print(f"   BASE_DIR: {BASE_DIR}")
                self.faq_data = {}
            
            # FAQ ã‚¢ã‚¤ãƒ†ãƒ ã‚’ãƒ•ãƒ©ãƒƒãƒˆåŒ–
            # å¯¾å¿œãƒ•ã‚©ãƒ¼ãƒžãƒƒãƒˆ:
            #   A) {"faqs": [...], "meta": {...}}  â† faqsé…åˆ—ç›´æŽ¥
            #   B) {"ã‚«ãƒ†ã‚´ãƒªå": [...], ...}       â† ã‚«ãƒ†ã‚´ãƒªè¾žæ›¸
            self.faq_items_flat = []

            # ãƒ•ã‚©ãƒ¼ãƒžãƒƒãƒˆA: "faqs" ã‚­ãƒ¼ã«é…åˆ—ãŒå…¥ã£ã¦ã„ã‚‹å ´åˆ
            if "faqs" in self.faq_data and isinstance(self.faq_data["faqs"], list):
                print(f"📋 Processing {len(self.faq_data['faqs'])} FAQ items from 'faqs' array")
                for item in self.faq_data["faqs"]:
                    if isinstance(item, dict):
                        # フィールド正規化: faq_id → id, answer_steps → steps
                        normalized_item = item.copy()
                        if "faq_id" in normalized_item and "id" not in normalized_item:
                            normalized_item["id"] = normalized_item.pop("faq_id")
                        if "answer_steps" in normalized_item and "steps" not in normalized_item:
                            normalized_item["steps"] = normalized_item.pop("answer_steps")
                        
                        # 不要なフィールドを削除
                        for field in ["manual_ref", "confidence", "support_based"]:
                            if field in normalized_item:
                                del normalized_item[field]
                        
                        # utterances がない場合は question で代用
                        if "utterances" not in normalized_item and "question" in normalized_item:
                            normalized_item["utterances"] = [normalized_item["question"]]
                        self.faq_items_flat.append(normalized_item)
                print(f"✅ Normalized {len(self.faq_items_flat)} FAQ items")
            else:
                # ãƒ•ã‚©ãƒ¼ãƒžãƒƒãƒˆB: ã‚«ãƒ†ã‚´ãƒªè¾žæ›¸å½¢å¼
                for category_key, items in self.faq_data.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                if "category" not in item:
                                    item["category"] = category_key
                                self.faq_items_flat.append(item)
            
            # ã‚³ãƒ¼ãƒ‘ã‚¹æ§‹ç¯‰ï¼ˆquestion / utterances / steps / keywords ã‚’çµ±åˆï¼‰
            # FAQ items合計のログ出力
            print(f"📊 Total FAQ items loaded: {len(self.faq_items_flat)}")
            
            self.faq_corpus = []
            for item in self.faq_items_flat:
                text_parts = [
                    item.get("question", ""),
                    " ".join(item.get("utterances", [])),
                    " ".join(item.get("steps", [])),
                    " ".join(item.get("keywords", [])),
                    " ".join(item.get("tags", [])),  # tags も検索対象に
                    item.get("intent", ""),
                    item.get("category", ""),
                ]
                combined = " ".join(filter(None, text_parts))
                self.faq_corpus.append(normalize_text(combined))
            
            # FAISS ã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹æ§‹ç¯‰
            if self.faq_corpus:
                self.faq_embeddings = self.model.encode(
                    self.faq_corpus,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                faiss.normalize_L2(self.faq_embeddings)
                self.faq_index = build_optimized_index(self.faq_embeddings)
            
            self.faq_loaded = True
            print(f"âœ… FAQ data loaded: {len(self.faq_items_flat)} items")

state = AppState()

# ============================================
# ãƒ¦ãƒ¼ãƒ†ã‚£ãƒªãƒ†ã‚£é–¢æ•°
# ============================================

@lru_cache(maxsize=1000)
def normalize_text(text: str) -> str:
    """ãƒ†ã‚­ã‚¹ãƒˆæ­£è¦åŒ–ï¼ˆã‚­ãƒ£ãƒƒã‚·ãƒ¥ä»˜ãï¼‰"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def expand_with_synonyms(query: str, synonyms: Dict[str, Any], language: str = "ja") -> str:
    """
    同義語展開（言語別対応）
    
    Args:
        query: 検索クエリ
        synonyms: 言語別同義語辞書 {"ja": {...}, "en": {...}}
        language: 言語コード（"ja", "en"）
    
    Returns:
        展開されたクエリ
    """
    expanded_terms = [query]
    
    # 言語別の同義語辞書を取得
    lang_synonyms = synonyms.get(language, {}) if isinstance(synonyms, dict) else {}
    
    # 同義語展開
    for term, syns in lang_synonyms.items():
        if term.lower() in query.lower():
            expanded_terms.extend(syns)
    
    return " ".join(expanded_terms)

def title_match_score(query: str, title: str) -> float:
    """
    タイトルへのキーワード一致度を計算（0.0～1.0）
    
    Args:
        query: 検索クエリ
        title: 動画タイトル
    
    Returns:
        一致度スコア（0.0～1.0）
    """
    if not query or not title:
        return 0.0
    
    query_words = normalize_text(query).split()
    title_norm = normalize_text(title)
    
    if not query_words:
        return 0.0
    
    # クエリの各ワードがタイトルに含まれるかチェック
    matched = sum(1 for word in query_words if word in title_norm)
    
    # 一致率を返す
    return matched / len(query_words)


def build_optimized_index(embeddings: np.ndarray) -> faiss.Index:
    """æœ€é©åŒ–ã•ã‚ŒãŸFAISSã‚¤ãƒ³ãƒ‡ãƒƒã‚¯ã‚¹æ§‹ç¯‰"""
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
    """æ¤œç´¢ãƒ­ã‚°è¨˜éŒ²"""
    try:
        with open(SEARCH_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(timezone.utc).isoformat(), query])
    except Exception as e:
        print(f"âš ï¸ Log write failed: {e}")

def parse_logs() -> List[Dict[str, Any]]:
    """ログパース（管理画面用）- search_logs.jsonを読み込む"""
    log_file = BASE_DIR / "search_logs.json"
    
    if not log_file.exists():
        return []
    
    rows = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
        
        for log in logs:
            try:
                # timestampをdatetimeに変換
                timestamp = log.get("timestamp", "")
                if timestamp:
                    # ISO形式のタイムスタンプをパース
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    rows.append({
                        "dt": dt,
                        "query": log.get("query", ""),
                        "result_type": log.get("result_type", ""),
                        "result_id": log.get("result_id", "")
                    })
            except Exception as e:
                # パースエラーは無視
                pass
    except Exception as e:
        print(f"❌ Failed to parse logs: {e}")
        return []
    
    return rows

# ============================================
# èªè¨¼
# ============================================

security = HTTPBasic()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """管理者認証 - users.jsonを参照"""
    try:
        # users.jsonを読み込み
        if not USERS_PATH.exists():
            print("❌ users.json not found")
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        
        with open(USERS_PATH, 'r', encoding='utf-8') as f:
            users_data = json.load(f)
        
        # ユーザーを検索
        user = next((u for u in users_data["users"] if u["username"] == credentials.username), None)
        
        if not user:
            print(f"❌ User not found: {credentials.username}")
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        
        # パスワードを確認
        if user["password"] != credentials.password:
            print(f"❌ Invalid password for user: {credentials.username}")
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        
        print(f"✅ Authentication successful: {credentials.username}")
        return credentials.username
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Authentication error: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

# ============================================
# Lifespanç®¡ç†
# ============================================

# ============================================
# 起動時の初期化処理
# ============================================

def initialize_files():
    """必要なファイルを初期化"""
    try:
        # config.json の初期化
        if not CONFIG_PATH.exists():
            print("📁 Creating config.json...")
            default_config = {
                "faq_search_enabled": True,
                "similarity_threshold": 0.3,
                "semantic_weight": 0.6,
                "title_weight": 0.4
            }
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            print("✅ config.json created")
        
        # data.json の初期化
        if not DATA_PATH.exists():
            print("📁 Creating data.json...")
            with open(DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump([], f)
            print("✅ data.json created")
        
        # synonyms.json の初期化（言語別構造）
        if not SYNONYMS_PATH.exists():
            print("📁 Creating synonyms.json...")
            default_synonyms = {
                "ja": {
                    "プロッタ": ["プロッター", "大判プリンタ"],
                    "パスワード": ["PW", "pass"]
                },
                "en": {
                    "plotter": ["large format printer"],
                    "password": ["pw", "pass"]
                }
            }
            with open(SYNONYMS_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_synonyms, f, ensure_ascii=False, indent=2)
            print("✅ synonyms.json created")
        
        # faq.json の初期化
        if not FAQ_PATH.exists():
            print("📁 Creating faq.json...")
            default_faq = {
                "meta": {"version": "1.0", "count": 0},
                "faqs": []
            }
            with open(FAQ_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_faq, f, ensure_ascii=False, indent=2)
            print("✅ faq.json created")
        
        # users.json の初期化
        if not USERS_PATH.exists():
            print("📁 Creating users.json...")
            default_users = {
                "users": [
                    {
                        "username": "admin",
                        "password": "admin",
                        "secret_question": "",
                        "secret_answer": ""
                    }
                ]
            }
            with open(USERS_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_users, f, ensure_ascii=False, indent=2)
            print("✅ users.json created")
        
        print("✅ File initialization completed")
        
    except Exception as e:
        print(f"❌ File initialization error: {e}")
        import traceback
        traceback.print_exc()

# 起動時に初期化を実行
initialize_files()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """èµ·å‹•æ™‚ã¯æœ€å°é™ã®åˆæœŸåŒ–ã®ã¿"""
    print("ðŸš€ Application starting...")
    
    # ãƒ­ã‚°ãƒ•ã‚¡ã‚¤ãƒ«åˆæœŸåŒ–
    if not SEARCH_LOG_PATH.exists():
        with open(SEARCH_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "query"])
    
    # ãƒ‡ã‚£ãƒ¬ã‚¯ãƒˆãƒªä½œæˆ
    admin_path.mkdir(parents=True, exist_ok=True)
    
    print("âœ… Application ready (lazy loading enabled)")
    yield
    print("ðŸ›‘ Application shutting down...")

# ============================================
# FastAPI ã‚¢ãƒ—ãƒªã‚±ãƒ¼ã‚·ãƒ§ãƒ³
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
# ãƒ˜ãƒ«ã‚¹ãƒã‚§ãƒƒã‚¯
# ============================================

@app.get("/health")
async def health_check():
    """ãƒ˜ãƒ«ã‚¹ãƒã‚§ãƒƒã‚¯ï¼ˆRender.comç”¨ï¼‰"""
    return {
        "status": "healthy",
        "model_loaded": state.model_loaded,
        "video_loaded": state.video_loaded,
        "faq_loaded": state.faq_loaded,
        "faq_items_count": len(state.faq_items_flat),
        "faq_corpus_count": len(state.faq_corpus),
        "faq_index_available": state.faq_index is not None,
        "video_items_count": len(state.video_items_flat),
    }

# ============================================
# å‹•ç”»æ¤œç´¢ã‚¨ãƒ³ãƒ‰ãƒã‚¤ãƒ³ãƒˆ
# ============================================

@app.get("/search")
async def search_videos(
    query: str = Query(..., min_length=1),
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    paged: int = Query(0),
    language: str = Query(None)
):
    """å‹•ç”»æ¤œç´¢ï¼ˆãƒšãƒ¼ã‚¸ãƒ³ã‚°å¯¾å¿œï¼‰"""
    await state.ensure_video_loaded()
    
    if not state.video_index:
        return {"items": [], "has_more": False, "total_visible": 0}
    
    log_search(f"video:{query}")
    
    normalized_query = normalize_text(query)
    expanded_query = expand_with_synonyms(normalized_query, state.synonyms, language or "ja")
    
    # 設定から閾値を取得
    config = await get_config()
    threshold = config.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
    print(f"🔍 Video search: query='{query}', threshold={threshold}, config={config}")
    
    query_embedding = state.model.encode([expanded_query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    # 最大取得件数を100件に拡大（質問2の対応）
    k = min(offset + limit + 100, len(state.videos))
    distances, indices = state.video_index.search(query_embedding, k)
    
    results = []
    # 設定から重み配分を取得
    semantic_weight = config.get("semantic_weight", SEMANTIC_WEIGHT)
    title_weight = config.get("title_weight", TITLE_WEIGHT)
    
    all_scores = []  # 全スコアを記録
    for idx, score in zip(indices[0], distances[0]):
        if 0 <= idx < len(state.videos):
            all_scores.append(float(score))
            video = state.videos[idx].copy()
            
            # 言語フィルタリング
            video_lang = video.get("language", "ja")  # デフォルトは日本語
            if language != "all" and video_lang != language:
                continue
            
            # ハイブリッドスコアリング
            semantic_score = float(score)
            title_bonus = title_match_score(query, video.get("title", ""))
            hybrid_score = semantic_score * semantic_weight + title_bonus * title_weight
            
            video["score"] = hybrid_score
            video["semantic_score"] = semantic_score
            video["title_score"] = title_bonus
            
            # 閾値チェック（セマンティックスコアで判定）
            if semantic_score >= threshold:
                results.append(video)
    
    # デバッグ: 全体のスコア分布を表示
    if all_scores:
        print(f"  Total candidates: {len(all_scores)}")
        print(f"  Score range: {min(all_scores):.3f} - {max(all_scores):.3f}")
        print(f"  Top 5 scores: {[f'{s:.3f}' for s in sorted(all_scores, reverse=True)[:5]]}")
        print(f"  Passed threshold ({threshold}): {len(results)} items")
    
    # タイトル完全一致を最優先、次点でハイブリッドスコアでソート
    query_norm = normalize_text(query)
    results.sort(key=lambda v: (
        -int(query_norm in normalize_text(v.get("title", ""))),  # タイトル一致を最優先
        -v["score"]  # 次点でハイブリッドスコア
    ))
    
    total = len(results)
    items = results[offset:offset + limit]
    has_more = (offset + limit) < total
    
    # デバッグ: スコアの範囲を確認
    if results:
        scores = [r["score"] for r in results[:10]]  # 上位10件のスコア
        print(f"🎬 Video search results: query='{query}', total={total}, items={len(items)}, threshold={threshold}")
        print(f"   Top 10 scores: {scores}")
    else:
        print(f"🎬 Video search results: query='{query}', total=0, items=0, threshold={SIMILARITY_THRESHOLD}")
    
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
# FAQæ¤œç´¢ã‚¨ãƒ³ãƒ‰ãƒã‚¤ãƒ³ãƒˆ
# ============================================

@app.get("/faq/search")
async def search_faq(
    query: str = Query(..., min_length=1),
    offset: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
    paged: int = Query(0),
    language: str = Query(None)
):
    """FAQæ¤œç´¢ï¼ˆãƒšãƒ¼ã‚¸ãƒ³ã‚°å¯¾å¿œï¼‰"""
    await state.ensure_faq_loaded()
    
    if not state.faq_index:
        # FAISSインデックスが使用できない場合のフォールバック検索（簡易テキストマッチング）
        print(f"⚠️ FAQ index not available, using fallback text search")
        print(f"   FAQ items available: {len(state.faq_items_flat)}")
        print(f"   Query: '{query}'")
        
        normalized_query = normalize_text(query)
        expanded_query = expand_with_synonyms(normalized_query, state.synonyms, language or "ja")
        print(f"   Normalized/expanded query: '{expanded_query}'")
        
        # クエリを単語に分割
        query_words = expanded_query.lower().split()
        print(f"   Query words: {query_words}")
        
        # 各FAQアイテムとのマッチングスコアを計算
        scored_items = []
        for item in state.faq_items_flat:
            # 検索対象テキストを構築
            search_text = " ".join([
                item.get("question", ""),
                " ".join(item.get("utterances", [])),
                " ".join(item.get("steps", [])),
                " ".join(item.get("keywords", [])),
                " ".join(item.get("tags", [])),
                item.get("category", ""),
            ]).lower()
            
            # マッチングスコアを計算（単語が含まれている数）
            score = sum(1 for word in query_words if word in search_text)
            
            if score > 0:
                item_copy = item.copy()
                item_copy["score"] = float(score)
                scored_items.append(item_copy)
        
        print(f"   Matched items: {len(scored_items)}")
        
        # スコア順にソート（降順）
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        
        total = len(scored_items)
        items = scored_items[offset:offset + limit]
        has_more = (offset + limit) < total
        
        if paged:
            return {
                "items": items,
                "has_more": has_more,
                "total_visible": total,
                "offset": offset,
                "limit": limit,
                "fallback": True
            }
        
        return {"items": items, "fallback": True}
    
    log_search(f"faq:{query}")
    
    # 設定から閾値を取得
    config = await get_config()
    threshold = config.get("similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
    print(f"🔍 FAQ search: query='{query}', threshold={threshold}, config={config}")
    
    normalized_query = normalize_text(query)
    # 同義語展開で検索精度向上（Bug#5対応）
    expanded_query = expand_with_synonyms(normalized_query, state.synonyms, language or "ja")
    query_embedding = state.model.encode([expanded_query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    
    # 最大取得件数を100件に拡大
    k = min(offset + limit + 100, len(state.faq_items_flat))
    distances, indices = state.faq_index.search(query_embedding, k)
    
    # 設定から重み配分を取得
    semantic_weight = config.get("semantic_weight", SEMANTIC_WEIGHT)
    title_weight = config.get("title_weight", TITLE_WEIGHT)
    
    results = []
    all_scores = []  # 全スコアを記録
    for idx, score in zip(indices[0], distances[0]):
        if 0 <= idx < len(state.faq_items_flat):
            all_scores.append(float(score))
            item = state.faq_items_flat[idx].copy()
            
            # 言語フィルタリング
            item_lang = item.get("language", "ja")  # デフォルトは日本語
            if language != "all" and item_lang != language:
                continue
            
            # ハイブリッドスコアリング（FAQは質問文をタイトル扱い）
            semantic_score = float(score)
            title_bonus = title_match_score(query, item.get("question", ""))
            hybrid_score = semantic_score * semantic_weight + title_bonus * title_weight
            
            item["score"] = hybrid_score
            item["semantic_score"] = semantic_score
            item["title_score"] = title_bonus
            
            # 閾値チェック
            if semantic_score >= threshold:
                results.append(item)
    
    # デバッグ: 全体のスコア分布を表示
    if all_scores:
        print(f"  Total candidates: {len(all_scores)}")
        print(f"  Score range: {min(all_scores):.3f} - {max(all_scores):.3f}")
        print(f"  Top 5 scores: {[f'{s:.3f}' for s in sorted(all_scores, reverse=True)[:5]]}")
        print(f"  Passed threshold ({threshold}): {len(results)} items")
    
    # 質問文完全一致を最優先、次点でハイブリッドスコアでソート
    query_norm = normalize_text(query)
    results.sort(key=lambda v: (
        -int(query_norm in normalize_text(v.get("question", ""))),  # 質問文一致を最優先
        -v["score"]  # 次点でハイブリッドスコア
    ))
    
    total = len(results)
    items = results[offset:offset + limit]
    has_more = (offset + limit) < total
    
    # デバッグ: スコアの範囲を確認
    if results:
        scores = [r["score"] for r in results[:10]]  # 上位10件のスコア
        print(f"🎬 Video search results: query='{query}', total={total}, items={len(items)}, threshold={threshold}")
        print(f"   Top 10 scores: {scores}")
    else:
        print(f"🎬 Video search results: query='{query}', total=0, items=0, threshold={SIMILARITY_THRESHOLD}")
    
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
# ç®¡ç†API - ãƒ‡ãƒ¼ã‚¿ç·¨é›†
# ============================================

@app.get("/admin/api/synonyms", dependencies=[Depends(verify_admin)])
async def get_synonyms():
    """åŒç¾©èªžè¾žæ›¸å–å¾—"""
    if SYNONYMS_PATH.exists():
        with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@app.put("/admin/api/synonyms", dependencies=[Depends(verify_admin)])
async def update_synonyms(data: dict):
    """同義語辞書更新"""
    print(f"💾 Saving synonyms: {len(data)} terms")
    with open(SYNONYMS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Synonyms saved successfully")
    return {"status": "ok", "count": len(data)}

async def reload_video_data():
    """å‹•ç”»ãƒ‡ãƒ¼ã‚¿ã®ãƒªãƒ­ãƒ¼ãƒ‰"""
    state.video_loaded = False
    await state.ensure_video_loaded()

@app.post("/admin/api/synonyms/generate", dependencies=[Depends(verify_admin)])
async def generate_synonyms(background_tasks: BackgroundTasks):
    """data.jsonã‹ã‚‰åŒç¾©èªžã‚’ç”Ÿæˆ"""
    await state.ensure_video_loaded()
    
    synonym_map = {}
    for v in state.videos:
        title = v.get("title", "")
        desc = v.get("description", "")
        
        # ã‚¿ã‚¤ãƒˆãƒ«ã‹ã‚‰ä¸»è¦ã‚­ãƒ¼ãƒ¯ãƒ¼ãƒ‰æŠ½å‡ºï¼ˆç°¡æ˜“ç‰ˆï¼‰
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
    """åŒç¾©èªžã®å€‹åˆ¥æ›´æ–°"""
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
async def delete_synonym_term(term: str):
    """同義語の個別削除"""
    print(f"🗑️ Deleting synonym: {term}")
    synonyms = {}
    if SYNONYMS_PATH.exists():
        with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
            synonyms = json.load(f)
    
    if term in synonyms:
        del synonyms[term]
    
    with open(SYNONYMS_PATH, "w", encoding="utf-8") as f:
        json.dump(synonyms, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Synonym deleted successfully")
    return {"status": "ok", "term": term}

@app.get("/admin/api/faq", dependencies=[Depends(verify_admin)])
async def get_faq():
    """FAQå…¨ä½“å–å¾—"""
    if FAQ_PATH.exists():
        with open(FAQ_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@app.put("/admin/api/faq", dependencies=[Depends(verify_admin)])
async def update_faq(data: dict, background_tasks: BackgroundTasks):
    """FAQå…¨ä½“æ›´æ–°"""
    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_faq_data)
    return {"status": "ok"}

async def reload_faq_data():
    """FAQãƒ‡ãƒ¼ã‚¿ã®ãƒªãƒ­ãƒ¼ãƒ‰"""
    state.faq_loaded = False
    await state.ensure_faq_loaded()

# ============================================
# ç®¡ç†API - FAQå€‹åˆ¥ç·¨é›†
# ============================================

@app.get("/admin/api/faq/items", dependencies=[Depends(verify_admin)])
async def list_faq_items(offset: int = 0, limit: int = 50, q: str = ""):
    """FAQä¸€è¦§å–å¾—ï¼ˆæ¤œç´¢ãƒ»ãƒšãƒ¼ã‚¸ãƒ³ã‚°å¯¾å¿œï¼‰"""
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
    """FAQæ–°è¦ä½œæˆ"""
    faq_id = item.get("id")
    if not faq_id:
        raise HTTPException(400, "ID is required")
    
    await state.ensure_faq_loaded()
    if any(f.get("id") == faq_id for f in state.faq_items_flat):
        raise HTTPException(400, f"ID '{faq_id}' already exists")
    
    category = item.get("category", "ãã®ä»–")
    if category not in state.faq_data:
        state.faq_data[category] = []
    
    state.faq_data[category].append(item)
    
    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        json.dump(state.faq_data, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_faq_data)
    return {"status": "created", "id": faq_id}

@app.patch("/admin/api/faq/item/{item_id}", dependencies=[Depends(verify_admin)])
async def update_faq_item(item_id: str, item: dict, background_tasks: BackgroundTasks):
    """FAQæ›´æ–°"""
    await state.ensure_faq_loaded()
    
    found = False
    # faqs配列形式（Bug#1修正）
    if "faqs" in state.faq_data and isinstance(state.faq_data["faqs"], list):
        for i, existing in enumerate(state.faq_data["faqs"]):
            if isinstance(existing, dict) and existing.get("id") == item_id:
                state.faq_data["faqs"][i] = item
                found = True
                break
    else:
        # カテゴリ辞書形式（後方互換）
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
    # faqs配列形式（Bug#1修正）
    if "faqs" in state.faq_data and isinstance(state.faq_data["faqs"], list):
        for i, existing in enumerate(state.faq_data["faqs"]):
            if isinstance(existing, dict) and existing.get("id") == item_id:
                del state.faq_data["faqs"][i]
                found = True
                break
    else:
        # カテゴリ辞書形式（後方互換）
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

@app.delete("/admin/api/faq/all", dependencies=[Depends(verify_admin)])
async def delete_all_faqs(background_tasks: BackgroundTasks):
    """FAQ一括削除"""
    print("🗑️ FAQ bulk delete requested")
    await state.ensure_faq_loaded()
    
    # faqs配列形式
    if "faqs" in state.faq_data and isinstance(state.faq_data["faqs"], list):
        count = len(state.faq_data["faqs"])
        state.faq_data["faqs"] = []
        print(f"✅ Deleted {count} FAQs (faqs array format)")
    else:
        # カテゴリ辞書形式
        count = sum(len(items) for items in state.faq_data.values() if isinstance(items, list))
        for key in list(state.faq_data.keys()):
            if isinstance(state.faq_data[key], list):
                state.faq_data[key] = []
        print(f"✅ Deleted {count} FAQs (category dict format)")
    
    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        json.dump(state.faq_data, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_faq_data)
    return {"status": "deleted", "count": count}

@app.post("/admin/api/faq/import", dependencies=[Depends(verify_admin)])
async def import_faqs(data: dict, background_tasks: BackgroundTasks):
    """FAQインポート（新規追加・更新のみ）"""
    print("📤 FAQ import requested")
    await state.ensure_faq_loaded()
    
    imported_faqs = data.get("faqs", [])
    if not isinstance(imported_faqs, list):
        print(f"❌ Invalid format: faqs is {type(imported_faqs)}")
        raise HTTPException(400, "Invalid format: 'faqs' must be an array")
    
    print(f"📋 Importing {len(imported_faqs)} FAQs")
    added_count = 0
    updated_count = 0
    
    # faqs配列形式
    if "faqs" in state.faq_data and isinstance(state.faq_data["faqs"], list):
        existing_ids = {item.get("id") for item in state.faq_data["faqs"] if isinstance(item, dict)}
        print(f"   Existing FAQ IDs: {len(existing_ids)}")
        
        for imported_item in imported_faqs:
            if not isinstance(imported_item, dict):
                continue
            
            item_id = imported_item.get("id") or imported_item.get("faq_id")
            if not item_id:
                print(f"   ⚠️ Skipping item without ID")
                continue
            
            # フィールド正規化
            normalized_item = imported_item.copy()
            if "faq_id" in normalized_item and "id" not in normalized_item:
                normalized_item["id"] = normalized_item.pop("faq_id")
            if "answer_steps" in normalized_item and "steps" not in normalized_item:
                normalized_item["steps"] = normalized_item.pop("answer_steps")
            
            if item_id in existing_ids:
                # 更新
                for i, existing in enumerate(state.faq_data["faqs"]):
                    if existing.get("id") == item_id:
                        state.faq_data["faqs"][i] = normalized_item
                        updated_count += 1
                        print(f"   ✏️ Updated: {item_id}")
                        break
            else:
                # 新規追加
                state.faq_data["faqs"].append(normalized_item)
                added_count += 1
                print(f"   ➕ Added: {item_id}")
    else:
        # カテゴリ辞書形式への対応
        for imported_item in imported_faqs:
            if not isinstance(imported_item, dict):
                continue
            
            item_id = imported_item.get("id")
            if not item_id:
                continue
            
            category = imported_item.get("category", "その他")
            
            # カテゴリが存在しない場合は作成
            if category not in state.faq_data:
                state.faq_data[category] = []
            
            # 既存チェック
            found = False
            if isinstance(state.faq_data[category], list):
                for i, existing in enumerate(state.faq_data[category]):
                    if existing.get("id") == item_id:
                        state.faq_data[category][i] = imported_item
                        updated_count += 1
                        found = True
                        break
            
            if not found:
                state.faq_data[category].append(imported_item)
                added_count += 1
    
    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        json.dump(state.faq_data, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_faq_data)
    return {
        "status": "imported",
        "added": added_count,
        "updated": updated_count,
        "total": added_count + updated_count
    }

@app.get("/admin/api/faq/export", dependencies=[Depends(verify_admin)])
async def export_faqs():
    """FAQエクスポート"""
    print("📥 FAQ export requested")
    await state.ensure_faq_loaded()
    
    # フィールド名を元に戻す関数
    def normalize_for_export(faq):
        """エクスポート用にフィールド名を元に戻す"""
        exported_faq = faq.copy()
        
        # id → faq_id
        if "id" in exported_faq:
            exported_faq["faq_id"] = exported_faq.pop("id")
        
        # steps → answer_steps
        if "steps" in exported_faq:
            exported_faq["answer_steps"] = exported_faq.pop("steps")
        
        return exported_faq
    
    # faqs配列形式で返す
    if "faqs" in state.faq_data and isinstance(state.faq_data["faqs"], list):
        # 各FAQのフィールド名を元に戻す
        exported_faqs = [normalize_for_export(faq) for faq in state.faq_data["faqs"]]
        
        export_data = {
            "meta": state.faq_data.get("meta", {}),
            "faqs": exported_faqs
        }
        print(f"✅ Exporting {len(exported_faqs)} FAQs (faqs array format)")
    else:
        # カテゴリ辞書形式をfaqs配列形式に変換
        all_faqs = []
        for category, items in state.faq_data.items():
            if isinstance(items, list):
                all_faqs.extend(items)
        
        # 各FAQのフィールド名を元に戻す
        exported_faqs = [normalize_for_export(faq) for faq in all_faqs]
        
        import time
        export_data = {
            "meta": {
                "exported_at": time.time(),
                "count": len(exported_faqs)
            },
            "faqs": exported_faqs
        }
        print(f"✅ Exporting {len(exported_faqs)} FAQs (category dict format)")
    
    return export_data


# ============================================
# ç®¡ç†API - å‹•ç”»ãƒ‡ãƒ¼ã‚¿
# ============================================

@app.get("/admin/api/videos", dependencies=[Depends(verify_admin)])
async def get_videos():
    """å‹•ç”»ãƒ‡ãƒ¼ã‚¿ä¸€è¦§å–å¾—"""
    if not DATA_PATH.exists():
        return []
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        videos = json.load(f)
    
    return videos

@app.post("/admin/api/videos", dependencies=[Depends(verify_admin)])
async def create_video(video_data: dict, background_tasks: BackgroundTasks):
    """å‹•ç”»ãƒ‡ãƒ¼ã‚¿ä½œæˆ"""
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
    """å‹•ç”»ãƒ‡ãƒ¼ã‚¿æ›´æ–°"""
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

@app.put("/admin/api/videos", dependencies=[Depends(verify_admin)])
async def update_videos(videos: List[dict], background_tasks: BackgroundTasks):
    """動画データ一括更新（言語タグ編集用）"""
    print(f"💾 Updating videos: {len(videos)} items")
    
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(videos, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Videos saved successfully")
    
    # FAISSインデックスを再構築
    background_tasks.add_task(reload_video_data)
    
    return {"status": "ok", "count": len(videos)}

@app.delete("/admin/api/videos/{video_id}", dependencies=[Depends(verify_admin)])
async def delete_video(video_id: str, background_tasks: BackgroundTasks):
    """å‹•ç”»ãƒ‡ãƒ¼ã‚¿å‰Šé™¤"""
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

@app.post("/admin/api/videos/bulk-delete", dependencies=[Depends(verify_admin)])
async def bulk_delete_videos(request_data: dict, background_tasks: BackgroundTasks):
    """å‹•ç”»ãƒ‡ãƒ¼ã‚¿ä¸€æ‹¬å‰Šé™¤"""
    video_ids = request_data.get("video_ids", [])
    
    if not video_ids:
        raise HTTPException(400, "video_ids is required")
    
    if not DATA_PATH.exists():
        raise HTTPException(404, "Data file not found")
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        videos = json.load(f)
    
    # å‰Šé™¤å¯¾è±¡ä»¥å¤–ã‚’æ®‹ã™
    filtered_videos = [v for v in videos if v.get('video_id') not in video_ids]
    
    # noç•ªå·ã‚’æŒ¯ã‚Šç›´ã—
    for i, video in enumerate(filtered_videos, 1):
        video['no'] = i
    
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(filtered_videos, f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    
    return {
        "status": "success",
        "deleted_count": len(video_ids),
        "remaining_count": len(filtered_videos)
    }

@app.post("/admin/api/videos/delete-all", dependencies=[Depends(verify_admin)])
@app.delete("/admin/api/videos/all", dependencies=[Depends(verify_admin)])
async def delete_all_videos(background_tasks: BackgroundTasks):
    """data.jsonå…¨å‰Šé™¤ï¼ˆå®Œå…¨ãƒªã‚»ãƒƒãƒˆï¼‰"""
    if not DATA_PATH.exists():
        # data.jsonãŒãªã„å ´åˆã‚‚ç©ºé…åˆ—ã‚’ä½œæˆ
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return {
            "status": "success",
            "message": "Data file created as empty array"
        }
    
    # ç©ºã®é…åˆ—ã§ä¸Šæ›¸ã
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    
    background_tasks.add_task(reload_video_data)
    
    return {
        "status": "success",
        "message": "All video data deleted"
    }

@app.post("/admin/api/videos/import", dependencies=[Depends(verify_admin)])
async def import_videos(import_data: dict, background_tasks: BackgroundTasks):
    """å‹•ç”»ãƒ‡ãƒ¼ã‚¿ã‚¤ãƒ³ãƒãƒ¼ãƒˆ"""
    mode = import_data.get("mode", "merge")
    new_data = import_data.get("data", [])
    
    if not isinstance(new_data, list):
        raise HTTPException(400, "Invalid data format")
    
    added = 0
    updated = 0
    
    if mode == "replace":
        # å…¨ä½“ç½®æ›
        with open(DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        added = len(new_data)
    else:
        # å·®åˆ†ãƒžãƒ¼ã‚¸
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

# reload_video_data defined above

# ============================================
# ç®¡ç†API - YouTubeæ–‡å­—èµ·ã“ã—
# ============================================

@app.post("/admin/api/youtube/fetch", dependencies=[Depends(verify_admin)])
async def fetch_youtube_videos(request_data: dict):
    """YouTubeãƒãƒ£ãƒ³ãƒãƒ«ã‹ã‚‰å‹•ç”»ãƒªã‚¹ãƒˆã‚’å–å¾—"""
    try:
        from googleapiclient.discovery import build
        
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise HTTPException(400, "YOUTUBE_API_KEY environment variable not set")
        
        channel_url = request_data.get("channel_url", "")
        max_results = request_data.get("max_results", 50)
        
        # ãƒãƒ£ãƒ³ãƒãƒ«IDã‚’æŠ½å‡º
        channel_id = None
        if "/c/" in channel_url or "/channel/" in channel_url or "/@" in channel_url:
            # ãƒãƒ£ãƒ³ãƒãƒ«åã‹ã‚‰IDã‚’å–å¾—ã™ã‚‹å¿…è¦ãŒã‚ã‚‹
            # ç°¡ç•¥åŒ–ã®ãŸã‚ã€ãƒ¦ãƒ¼ã‚¶ãƒ¼ã«ãƒãƒ£ãƒ³ãƒãƒ«IDã‚’ç›´æŽ¥å…¥åŠ›ã—ã¦ã‚‚ã‚‰ã†æ–¹å¼ã‚‚æ¤œè¨Ž
            parts = channel_url.rstrip('/').split('/')
            channel_name = parts[-1]
            
            youtube = build('youtube', 'v3', developerKey=api_key)
            
            # ãƒãƒ£ãƒ³ãƒãƒ«åã‹ã‚‰æ¤œç´¢
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
        
        # ãƒãƒ£ãƒ³ãƒãƒ«ã®ã‚¢ãƒƒãƒ—ãƒ­ãƒ¼ãƒ‰ãƒ—ãƒ¬ã‚¤ãƒªã‚¹ãƒˆIDã‚’å–å¾—
        youtube = build('youtube', 'v3', developerKey=api_key)
        channel_response = youtube.channels().list(
            id=channel_id,
            part='contentDetails'
        ).execute()
        
        if not channel_response.get('items'):
            raise HTTPException(404, "Channel not found")
        
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # ãƒ—ãƒ¬ã‚¤ãƒªã‚¹ãƒˆã‹ã‚‰å‹•ç”»ã‚’å–å¾—
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
    """YouTubeå‹•ç”»ã‚’æ–‡å­—èµ·ã“ã—"""
    try:
        video_id = request_data.get("video_id")
        if not video_id:
            raise HTTPException(400, "video_id is required")
        
        # æ—¢å­˜ã®data.jsonã‚’èª­ã¿è¾¼ã¿
        existing_videos = []
        if DATA_PATH.exists():
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                existing_videos = json.load(f)
        
        # æ—¢ã«å­˜åœ¨ã™ã‚‹ã‹ãƒã‚§ãƒƒã‚¯
        for video in existing_videos:
            if video.get('video_id') == video_id:
                return {
                    'status': 'already_exists',
                    'message': 'Video already transcribed',
                    'video_id': video_id
                }
        
        # ãƒãƒƒã‚¯ã‚°ãƒ©ã‚¦ãƒ³ãƒ‰ã§æ–‡å­—èµ·ã“ã—å‡¦ç†
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
    """YouTubeãƒãƒ£ãƒ³ãƒãƒ«ã¨data.jsonã‚’åŒæœŸï¼ˆå·®åˆ†æ¤œå‡ºï¼‰"""
    try:
        from googleapiclient.discovery import build
        
        api_key = os.getenv("YOUTUBE_API_KEY")
        if not api_key:
            raise HTTPException(400, "YOUTUBE_API_KEY environment variable not set")
        
        channel_url = request_data.get("channel_url", "")
        
        # ãƒãƒ£ãƒ³ãƒãƒ«IDã‚’æŠ½å‡º
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
        
        # ãƒãƒ£ãƒ³ãƒãƒ«ã®å…¨å‹•ç”»ã‚’å–å¾—
        youtube = build('youtube', 'v3', developerKey=api_key)
        channel_response = youtube.channels().list(
            id=channel_id,
            part='contentDetails'
        ).execute()
        
        if not channel_response.get('items'):
            raise HTTPException(404, "Channel not found")
        
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # ãƒ—ãƒ¬ã‚¤ãƒªã‚¹ãƒˆã‹ã‚‰å…¨å‹•ç”»ã‚’å–å¾—
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
        
        # æ—¢å­˜ã®data.jsonã‚’èª­ã¿è¾¼ã¿
        existing_videos = []
        if DATA_PATH.exists():
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                existing_videos = json.load(f)
        
        # å·®åˆ†ã‚’è¨ˆç®—
        youtube_ids = set(v['video_id'] for v in youtube_videos)
        existing_ids = set(v.get('video_id') for v in existing_videos)
        
        # YouTubeã«ã‚ã‚‹ãŒã€data.jsonã«ãªã„ï¼ˆè¿½åŠ ã™ã¹ãå‹•ç”»ï¼‰
        missing_in_data = [v for v in youtube_videos if v['video_id'] not in existing_ids]
        
        # data.jsonã«ã‚ã‚‹ãŒã€YouTubeã«ãªã„ï¼ˆå‰Šé™¤ã™ã¹ãå‹•ç”»ï¼‰
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
    """YouTubeã«å­˜åœ¨ã—ãªã„å‹•ç”»ã‚’data.jsonã‹ã‚‰å‰Šé™¤"""
    try:
        video_ids_to_delete = request_data.get("video_ids", [])
        
        if not video_ids_to_delete:
            raise HTTPException(400, "video_ids is required")
        
        if not DATA_PATH.exists():
            raise HTTPException(404, "Data file not found")
        
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            videos = json.load(f)
        
        # å‰Šé™¤å¯¾è±¡ä»¥å¤–ã®å‹•ç”»ã‚’æ®‹ã™
        filtered_videos = [v for v in videos if v.get('video_id') not in video_ids_to_delete]
        
        # noç•ªå·ã‚’æŒ¯ã‚Šç›´ã—
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
    """æ–‡å­—èµ·ã“ã—å‡¦ç†ï¼ˆãƒãƒƒã‚¯ã‚°ãƒ©ã‚¦ãƒ³ãƒ‰ï¼‰- yt-dlp Pythonãƒ©ã‚¤ãƒ–ãƒ©ãƒªä½¿ç”¨"""
    import tempfile
    import os
    import yt_dlp
    
    audio_path = None
    
    try:
        print(f"[INFO] Starting transcription for {video_id}: {video_data.get('title', '')}")
        
        # 1. éŸ³å£°ãƒ€ã‚¦ãƒ³ãƒ­ãƒ¼ãƒ‰ï¼ˆyt-dlp Pythonãƒ©ã‚¤ãƒ–ãƒ©ãƒªã‚’ä½¿ç”¨ï¼‰
        temp_dir = tempfile.gettempdir()
        output_basename = f"audio_{video_id}"
        output_path = os.path.join(temp_dir, output_basename)
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        
        print(f"[INFO] Downloading audio from {video_url}")
        
        # yt-dlpè¨­å®šï¼ˆbotæ¤œå‡ºå›žé¿ã‚’å«ã‚€ï¼‰
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path + '.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            # botæ¤œå‡ºå›žé¿: iOSã‚¯ãƒ©ã‚¤ã‚¢ãƒ³ãƒˆã‚’ä½¿ç”¨
            'extractor_args': {
                'youtube': {
                    'player_client': ['ios', 'android', 'web']
                }
            },
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'postprocessor_args': ['-ar', '16000'],
            'prefer_ffmpeg': True,
        }
        
        # yt-dlpã§ãƒ€ã‚¦ãƒ³ãƒ­ãƒ¼ãƒ‰
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] yt-dlp download failed: {error_msg}")
            
            # botæ¤œå‡ºã‚¨ãƒ©ãƒ¼ã®åˆ¤å®š
            if 'Sign in to confirm' in error_msg or 'bot' in error_msg.lower():
                raise Exception(
                    "YouTube botæ¤œå‡º: ã“ã®å‹•ç”»ã¯ç¾åœ¨ãƒ€ã‚¦ãƒ³ãƒ­ãƒ¼ãƒ‰ã§ãã¾ã›ã‚“ã€‚"
                )
            
            raise Exception(f"yt-dlp download failed: {error_msg[:300]}")
        
        # MP3ãƒ•ã‚¡ã‚¤ãƒ«ã®å­˜åœ¨ç¢ºèª
        audio_path = output_path + '.mp3'
        if not os.path.exists(audio_path):
            raise Exception(f"Audio file not created: {audio_path}")
        
        print(f"[INFO] Audio downloaded to {audio_path}")
        
        # 2. Whisperã§æ–‡å­—èµ·ã“ã—
        try:
            import whisper
            print(f"[INFO] Loading Whisper model...")
            model = whisper.load_model("base")
            print(f"[INFO] Transcribing...")
            result = model.transcribe(audio_path, language='ja', fp16=False)
            transcript = result['text']
            print(f"[INFO] Transcription completed: {len(transcript)} characters")
        except ImportError:
            raise Exception("Whisper not installed. Please install: pip install openai-whisper")
        except Exception as e:
            raise Exception(f"Whisper transcription failed: {str(e)}")
        
        # 3. data.jsonã«è¿½åŠ 
        existing_videos = []
        if DATA_PATH.exists():
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                existing_videos = json.load(f)
        
        # æ–°ã—ã„noç•ªå·ã‚’ç”Ÿæˆ
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
        
        print(f"[SUCCESS] Transcription saved for {video_id}")
        
        # å‹•ç”»ãƒ‡ãƒ¼ã‚¿å†èª­ã¿è¾¼ã¿
        await reload_video_data()
        
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Transcription error for {video_id}: {error_msg}")
        
        # ã‚¨ãƒ©ãƒ¼æ™‚ã‚‚data.jsonã«è¨˜éŒ²ï¼ˆstatus: failedï¼‰
        try:
            existing_videos = []
            if DATA_PATH.exists():
                with open(DATA_PATH, "r", encoding="utf-8") as f:
                    existing_videos = json.load(f)
            
            max_no = max([v.get('no', 0) for v in existing_videos], default=0)
            
            # ã‚¨ãƒ©ãƒ¼ãƒ¡ãƒƒã‚»ãƒ¼ã‚¸ã‚’åˆ†ã‹ã‚Šã‚„ã™ãå¤‰æ›
            friendly_error = error_msg
            if 'bot' in error_msg.lower() or 'Sign in to confirm' in error_msg:
                friendly_error = "YouTube botæ¤œå‡º: ã“ã®å‹•ç”»ã¯ç¾åœ¨ãƒ€ã‚¦ãƒ³ãƒ­ãƒ¼ãƒ‰ã§ãã¾ã›ã‚“ã€‚ã—ã°ã‚‰ãå¾…ã£ã¦ã‹ã‚‰å†è©¦è¡Œã—ã¦ãã ã•ã„ã€‚"
            elif 'JavaScript runtime' in error_msg:
                friendly_error = "JavaScriptå‡¦ç†ã‚¨ãƒ©ãƒ¼: ã“ã®å‹•ç”»ã¯ç‰¹æ®Šãªå‡¦ç†ãŒå¿…è¦ã§ã™ã€‚YouTube Data APIã‹ã‚‰å–å¾—ã—ãŸå‹•ç”»æƒ…å ±ã®ã¿ä¿å­˜ã•ã‚Œã¾ã™ã€‚"
            
            error_video = {
                'no': max_no + 1,
                'video_id': video_id,
                'title': video_data.get('title', ''),
                'description': video_data.get('description', ''),
                'transcript': f'æ–‡å­—èµ·ã“ã—å¤±æ•—: {friendly_error}',
                'url': video_data.get('url', ''),
                'thumbnail': video_data.get('thumbnail', ''),
                'status': 'failed'
            }
            
            existing_videos.append(error_video)
            
            with open(DATA_PATH, "w", encoding="utf-8") as f:
                json.dump(existing_videos, f, ensure_ascii=False, indent=2)
            
            print(f"[INFO] Error status saved for {video_id}")
        except Exception as save_error:
            print(f"[ERROR] Failed to save error status: {str(save_error)}")
    
    finally:
        # ä¸€æ™‚ãƒ•ã‚¡ã‚¤ãƒ«å‰Šé™¤
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"[INFO] Temporary file removed: {audio_path}")
            except Exception as cleanup_error:
                print(f"[WARNING] Failed to remove temp file: {str(cleanup_error)}")

# ============================================
# ç®¡ç†API - ãƒ­ã‚°
# ============================================

@app.get("/admin/api/logs/months", dependencies=[Depends(verify_admin)])
async def get_log_months():
    """åˆ©ç”¨å¯èƒ½ãªæœˆä¸€è¦§"""
    rows = parse_logs()
    months = set(r["dt"].strftime("%Y-%m") for r in rows)
    return {"months": sorted(months, reverse=True)}

@app.get("/admin/api/logs/summary", dependencies=[Depends(verify_admin)])
async def get_log_summary(month: str = Query(...)):
    """æœˆåˆ¥ã‚µãƒžãƒªãƒ¼"""
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
    """ãƒ­ã‚°CSVã‚¨ã‚¯ã‚¹ãƒãƒ¼ãƒˆ"""
    if not SEARCH_LOG_PATH.exists():
        raise HTTPException(404, "No logs found")
    
    csv_data = SEARCH_LOG_PATH.read_text(encoding="utf-8")
    headers = {"Content-Disposition": 'attachment; filename="search_logs.csv"'}
    return StreamingResponse(iter([csv_data]), media_type="text/csv", headers=headers)

# ============================================
# é™çš„ãƒ•ã‚¡ã‚¤ãƒ«é…ä¿¡
# ============================================

# ãƒ•ãƒ­ãƒ³ãƒˆã‚¨ãƒ³ãƒ‰é™çš„ãƒ•ã‚¡ã‚¤ãƒ«
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# ç®¡ç†ç”»é¢é™çš„ãƒ•ã‚¡ã‚¤ãƒ«
app.mount("/admin/static", StaticFiles(directory=admin_path), name="admin_static")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_index():
    """æ¤œç´¢ç”»é¢"""
    index_file = frontend_path / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return index_file.read_text(encoding="utf-8")


@app.get("/debug", response_class=HTMLResponse, include_in_schema=False)
def serve_debug():
    """診断ページ"""
    debug_file = frontend_path / "debug.html"
    if not debug_file.exists():
        return HTMLResponse("<h1>debug.html not found</h1>", status_code=404)
    return debug_file.read_text(encoding="utf-8")

@app.get("/admin", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_home():
    """ç®¡ç†ç”»é¢ãƒˆãƒƒãƒ—"""
    f = admin_path / "index.html"
    if not f.exists():
        return HTMLResponse("<h1>admin index.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/dashboard", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_dashboard():
    """ç®¡ç†ç”»é¢ - ãƒ€ãƒƒã‚·ãƒ¥ãƒœãƒ¼ãƒ‰"""
    f = admin_path / "dashboard.html"
    if not f.exists():
        return HTMLResponse("<h1>admin dashboard.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/videos", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_videos():
    """ç®¡ç†ç”»é¢ - å‹•ç”»ãƒ‡ãƒ¼ã‚¿"""
    f = admin_path / "videos.html"
    if not f.exists():
        return HTMLResponse("<h1>admin videos.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/synonyms", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_synonyms():
    """ç®¡ç†ç”»é¢ - Synonyms"""
    f = admin_path / "synonyms.html"
    if not f.exists():
        return HTMLResponse("<h1>admin synonyms.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/faq", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_faq():
    """ç®¡ç†ç”»é¢ - FAQ"""
    f = admin_path / "faq.html"
    if not f.exists():
        return HTMLResponse("<h1>admin faq.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/logs", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_logs():
    """ç®¡ç†ç”»é¢ - ãƒ­ã‚°"""
    f = admin_path / "logs.html"
    if not f.exists():
        return HTMLResponse("<h1>admin logs.html not found</h1>", status_code=404)

@app.get("/admin/password", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_password():
    """管理画面 - パスワード変更"""
    f = admin_path / "password.html"
    if not f.exists():
        return HTMLResponse("<h1>admin password.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/reset", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_reset():
    """管理画面 - パスワード再設定"""
    f = admin_path / "reset.html"
    if not f.exists():
        return HTMLResponse("<h1>admin reset.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

    return f.read_text(encoding="utf-8")

# .htmlæ‹¡å¼µå­ä»˜ãã®ãƒ«ãƒ¼ãƒˆã‚‚è¿½åŠ 
@app.get("/admin/dashboard.html", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_dashboard_html():
    """ç®¡ç†ç”»é¢ - ãƒ€ãƒƒã‚·ãƒ¥ãƒœãƒ¼ãƒ‰ (.html)"""
    f = admin_path / "dashboard.html"
    if not f.exists():
        return HTMLResponse("<h1>admin dashboard.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/videos.html", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_videos_html():
    """ç®¡ç†ç”»é¢ - å‹•ç”»ãƒ‡ãƒ¼ã‚¿ (.html)"""
    f = admin_path / "videos.html"
    if not f.exists():
        return HTMLResponse("<h1>admin videos.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/synonyms.html", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_synonyms_html():
    """ç®¡ç†ç”»é¢ - Synonyms (.html)"""
    f = admin_path / "synonyms.html"
    if not f.exists():
        return HTMLResponse("<h1>admin synonyms.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/faq.html", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_faq_html():
    """ç®¡ç†ç”»é¢ - FAQ (.html)"""
    f = admin_path / "faq.html"
    if not f.exists():
        return HTMLResponse("<h1>admin faq.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

@app.get("/admin/logs.html", response_class=HTMLResponse, include_in_schema=False)
def serve_admin_logs_html():
    """ç®¡ç†ç”»é¢ - ãƒ­ã‚° (.html)"""
    f = admin_path / "logs.html"
    if not f.exists():
        return HTMLResponse("<h1>admin logs.html not found</h1>", status_code=404)
    return f.read_text(encoding="utf-8")

# ============================================
# æ¤œç´¢ãƒ­ã‚°APIãƒ»Synonyms APIï¼ˆè¿½åŠ ï¼‰
# ============================================

@app.post("/api/log_search")
async def log_search_api(log_data: dict):
    """æ¤œç´¢ãƒ­ã‚°ã‚’è¨˜éŒ²"""
    import json
    from pathlib import Path
    
    log_file = Path("search_logs.json")
    logs = []
    
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except:
            logs = []
    
    logs.append({
        'query': log_data.get('query'),
        'result_type': log_data.get('result_type'),
        'result_id': log_data.get('result_id'),
        'timestamp': log_data.get('timestamp')
    })
    
    # æœ€æ–°1000ä»¶ã®ã¿ä¿æŒ
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs[-1000:], f, ensure_ascii=False, indent=2)
    
    return {'status': 'logged'}

@app.get("/api/ranking/faq")
async def get_faq_ranking(limit: int = 10):
    """FAQクリックランキング"""
    from pathlib import Path
    from collections import Counter
    
    log_file = Path("search_logs.json")
    if not log_file.exists():
        return {"ranking": []}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    except:
        return {"ranking": []}
    
    # FAQのクリックを集計
    faq_clicks = [log['result_id'] for log in logs if log.get('result_type') == 'faq' and log.get('result_id')]
    counter = Counter(faq_clicks)
    
    # FAQの詳細情報を取得
    await state.ensure_faq_loaded()
    
    ranking = []
    for faq_id, count in counter.most_common(limit):
        # FAQ情報を検索
        faq_item = next((item for item in state.faq_items_flat if item.get('id') == faq_id), None)
        if faq_item:
            ranking.append({
                'id': faq_id,
                'question': faq_item.get('question', ''),
                'category': faq_item.get('category', ''),
                'click_count': count
            })
    
    return {"ranking": ranking}

@app.get("/api/ranking/video")
async def get_video_ranking(limit: int = 10):
    """動画クリックランキング"""
    from pathlib import Path
    from collections import Counter
    
    log_file = Path("search_logs.json")
    if not log_file.exists():
        return {"ranking": []}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    except:
        return {"ranking": []}
    
    # 動画のクリックを集計
    video_clicks = [log['result_id'] for log in logs if log.get('result_type') == 'video' and log.get('result_id')]
    counter = Counter(video_clicks)
    
    # 動画の詳細情報を取得
    await state.ensure_video_loaded()
    
    ranking = []
    for video_id, count in counter.most_common(limit):
        # 動画情報を検索
        video_item = next((item for item in state.videos if item.get('video_id') == video_id), None)
        if video_item:
            ranking.append({
                'video_id': video_id,
                'title': video_item.get('title', ''),
                'thumbnail': video_item.get('thumbnail', ''),
                'click_count': count
            })
    
    return {"ranking": ranking}


# ============================================
# ユーザー管理API
# ============================================

@app.get("/admin/api/user", dependencies=[Depends(verify_admin)])
async def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    """現在のユーザー情報を取得"""
    try:
        with open(USERS_PATH, 'r', encoding='utf-8') as f:
            users_data = json.load(f)
        
        username = credentials.username
        user = next((u for u in users_data["users"] if u["username"] == username), None)
        
        if user:
            return {
                "username": user["username"],
                "has_secret_question": bool(user.get("secret_question"))
            }
        
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        print(f"❌ Get user error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/admin/api/user/password", dependencies=[Depends(verify_admin)])
async def change_password(request: dict, credentials: HTTPBasicCredentials = Depends(security)):
    """パスワード変更"""
    print(f"🔐 Password change requested for user: {credentials.username}")
    
    try:
        old_password = request.get("old_password")
        new_password = request.get("new_password")
        secret_question = request.get("secret_question", "")
        secret_answer = request.get("secret_answer", "")
        
        if not old_password or not new_password:
            raise HTTPException(status_code=400, detail="Old and new passwords are required")
        
        # ユーザー情報を読み込み
        with open(USERS_PATH, 'r', encoding='utf-8') as f:
            users_data = json.load(f)
        
        # ユーザーを検索
        user = next((u for u in users_data["users"] if u["username"] == credentials.username), None)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # 現在のパスワードを確認
        if user["password"] != old_password:
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        
        # パスワードと秘密の質問を更新
        user["password"] = new_password
        if secret_question:
            user["secret_question"] = secret_question
            user["secret_answer"] = secret_answer
        
        # 保存
        with open(USERS_PATH, 'w', encoding='utf-8') as f:
            json.dump(users_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Password changed successfully for user: {credentials.username}")
        return {"status": "ok", "message": "Password changed successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Password change error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/user/verify-answer")
async def verify_secret_answer(request: dict):
    """秘密の質問の回答を確認"""
    try:
        username = request.get("username")
        secret_answer = request.get("secret_answer")
        
        if not username or not secret_answer:
            raise HTTPException(status_code=400, detail="Username and answer are required")
        
        # ユーザー情報を読み込み
        with open(USERS_PATH, 'r', encoding='utf-8') as f:
            users_data = json.load(f)
        
        # ユーザーを検索
        user = next((u for u in users_data["users"] if u["username"] == username), None)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # 秘密の質問が設定されているか確認
        if not user.get("secret_question"):
            raise HTTPException(status_code=400, detail="Secret question not set")
        
        # 回答を確認
        if user["secret_answer"] == secret_answer:
            return {
                "status": "ok",
                "message": "Answer verified",
                "secret_question": user["secret_question"]
            }
        else:
            raise HTTPException(status_code=400, detail="Incorrect answer")
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Verify answer error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/user/reset-password")
async def reset_password(request: dict):
    """パスワード再設定"""
    print(f"🔐 Password reset requested")
    
    try:
        username = request.get("username")
        secret_answer = request.get("secret_answer")
        new_password = request.get("new_password")
        
        if not username or not secret_answer or not new_password:
            raise HTTPException(status_code=400, detail="All fields are required")
        
        # ユーザー情報を読み込み
        with open(USERS_PATH, 'r', encoding='utf-8') as f:
            users_data = json.load(f)
        
        # ユーザーを検索
        user = next((u for u in users_data["users"] if u["username"] == username), None)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # 秘密の質問が設定されているか確認
        if not user.get("secret_question"):
            raise HTTPException(status_code=400, detail="Secret question not set")
        
        # 回答を確認
        if user["secret_answer"] != secret_answer:
            raise HTTPException(status_code=400, detail="Incorrect answer")
        
        # パスワードを更新
        user["password"] = new_password
        
        # 保存
        with open(USERS_PATH, 'w', encoding='utf-8') as f:
            json.dump(users_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Password reset successfully for user: {username}")
        return {"status": "ok", "message": "Password reset successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Password reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/user/secret-question")
async def get_secret_question(username: str):
    """秘密の質問を取得"""
    try:
        with open(USERS_PATH, 'r', encoding='utf-8') as f:
            users_data = json.load(f)
        
        user = next((u for u in users_data["users"] if u["username"] == username), None)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        if not user.get("secret_question"):
            raise HTTPException(status_code=400, detail="Secret question not set")
        
        return {"secret_question": user["secret_question"]}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Get secret question error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/synonyms")
async def get_synonyms():
    """Synonyms.jsonã‚’è¿”ã™"""
    try:
        with open('synonyms.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Synonymsèª­ã¿è¾¼ã¿ã‚¨ãƒ©ãƒ¼: {e}")
        return {}


# ============================================
# 設定API
# ============================================

@app.get("/api/config")
async def get_config():
    """設定を取得"""
    print("📋 Config read requested")
    
    if not CONFIG_PATH.exists():
        # デフォルト設定を作成
        default_config = {
            "faq_search_enabled": True,
            "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
            "semantic_weight": SEMANTIC_WEIGHT,
            "title_weight": TITLE_WEIGHT
        }
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            print(f"✅ Created default config.json")
        except Exception as e:
            print(f"❌ Failed to create config.json: {e}")
        return default_config
    
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ Config loaded: {config}")
        return config
    except Exception as e:
        print(f"❌ Config read error: {e}")
        return {
            "faq_search_enabled": True,
            "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
            "semantic_weight": SEMANTIC_WEIGHT,
            "title_weight": TITLE_WEIGHT
        }

@app.get("/admin/api/config", dependencies=[Depends(verify_admin)])
async def get_admin_config():
    """設定を取得（管理画面用）"""
    return await get_config()


@app.put("/admin/api/config", dependencies=[Depends(verify_admin)])
async def update_config(config_data: dict):
    """設定を更新"""
    print(f"⚙️ Updating config: {config_data}")
    
    try:
        # config.jsonに書き込み
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        # 書き込み確認
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
        
        print(f"✅ Config saved successfully: {saved_config}")
        return {"status": "ok", "config": saved_config}
    except Exception as e:
        print(f"❌ Config update error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    try:
        port = int(os.getenv("PORT", 8000))
        print(f"🚀 Starting server on port {port}")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        raise
