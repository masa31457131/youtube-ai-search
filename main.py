
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import requests
import os
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
import pathlib

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
CHANNEL_ID = os.getenv("CHANNEL_ID")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
video_data = []
index = None

def fetch_youtube_videos():
    global video_data
    video_data.clear()

    url = f"https://www.googleapis.com/youtube/v3/search?key={API_KEY}&channelId={CHANNEL_ID}&part=snippet&type=video&maxResults=50"
    res = requests.get(url)
    data = res.json()

    if "error" in data:
        raise RuntimeError(f"YouTube APIエラー: {data['error']['message']}")

    items = data.get("items", [])
    if not items:
        raise RuntimeError("動画が取得できませんでした。APIキーまたはチャンネルIDが正しいか確認してください。")

    for item in items:
        snippet = item["snippet"]
        title = snippet["title"]
        description = snippet["description"]
        video_id = item["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        thumbnail = snippet["thumbnails"]["medium"]["url"]
        video_data.append((title, description, video_url, thumbnail))

def build_search_index():
    global index
    texts = [f"{title}. {desc}" for title, desc, _, _ in video_data]
    embeddings = model.encode(texts, convert_to_numpy=True)

    if len(embeddings) == 0:
        raise RuntimeError("動画がありません。データ取得に失敗しています。")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

@app.on_event("startup")
def startup_event():
    print("📺 YouTube動画取得中...")
    fetch_youtube_videos()
    build_search_index()
    print(f"✅ 動画数: {len(video_data)} 件取得・検索準備完了")

@app.get("/search")
def search(query: str = Query(..., description="検索キーワード")):
    q_embedding = model.encode([query])
    D, I = index.search(q_embedding, k=10)
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

frontend_path = pathlib.Path(__file__).parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
