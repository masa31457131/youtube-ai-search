from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import requests
import os
from dotenv import load_dotenv

from fastapi.staticfiles import StaticFiles
import pathlib

# .envの読み込み（ローカルまたはRender環境変数）
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
CHANNEL_ID = os.getenv("CHANNEL_ID")

# FastAPI アプリ本体
app = FastAPI()

# CORS設定（JSとの通信許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要なら限定してもOK
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデル・データ・インデックスの定義
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
video_data = []  # [(title, description, url, thumbnail), ...]
index = None     # FAISS index


# YouTube動画をAPIから取得
def fetch_youtube_videos():
    global video_data
    video_data.clear()
    next_page = ""
    while True:
        url = f"https://www.googleapis.com/youtube/v3/search?key={API_KEY}&channelId={CHANNEL_ID}&part=snippet&type=video&maxResults=50&pageToken={next_page}"
        res = requests.get(url)
        data = res.json()

        # APIキー・チャンネルIDが不正な場合の防止
        if "error" in data:
            raise RuntimeError(f"YouTube API error: {data['error']['message']}")

        for item in data.get("items", []):
            snippet = item["snippet"]
            title = snippet["title"]
            description = snippet["description"]
            video_id = item["id"]["videoId"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            thumbnail = snippet["thumbnails"]["medium"]["url"]
            video_data.append((title, description, video_url, thumbnail))

        next_page = data.get("nextPageToken", "")
        if not next_page:
            break


# 検索用インデックスを構築
def build_search_index():
    global index
    texts = [f"{title}. {desc}" for title, desc, _, _ in video_data]
    embeddings = model.encode(texts, convert_to_numpy=True)
    if len(embeddings) == 0:
        raise RuntimeError("動画が取得できていません。APIキーやチャンネルIDを確認してください。")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)


# アプリ起動時にデータ読み込み＆インデックス構築
@app.on_event("startup")
def startup_event():
    print("Fetching YouTube data...")
    fetch_youtube_videos()
    build_search_index()
    print(f"Loaded {len(video_data)} videos")


# 検索エンドポイント
@app.get("/search")
def search(query: str = Query(..., description="検索キーワード")):
    q_embedding = model.encode([query])
    D, I = index.search(q_embedding, k=10)  # 上位10件
    results = []
    for idx in I[0]:
        if idx < len(video_data):
            title, desc, url, thumbnail = video_data[idx]
            results.append({
                "title": title,
                "description": desc,
                "url": url,
                "thumbnail": thumbnail
            })
    return results


# フロントエンド（index.html）をルートで配信
frontend_path = pathlib.Path(__file__).parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
