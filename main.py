
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import os
import pathlib
import whisper
import requests
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from pytube import YouTube

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
whisper_model = whisper.load_model("base")
video_data = []
index = None

def download_and_transcribe(video_id):
    yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_path = f"audio_{video_id}.mp4"
    audio_stream.download(filename=audio_path)
    result = whisper_model.transcribe(audio_path, language="ja")
    os.remove(audio_path)
    return result["text"]

def fetch_youtube_videos():
    global video_data
    video_data.clear()

    url = f"https://www.googleapis.com/youtube/v3/search?key={API_KEY}&channelId={CHANNEL_ID}&part=snippet&type=video&maxResults=5"
    res = requests.get(url)
    data = res.json()

    if "error" in data:
        raise RuntimeError(f"YouTube APIエラー: {data['error']['message']}")

    items = data.get("items", [])
    for item in items:
        snippet = item["snippet"]
        title = snippet["title"]
        description = snippet["description"]
        video_id = item["id"]["videoId"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        thumbnail = snippet["thumbnails"]["medium"]["url"]
        try:
            transcript = download_and_transcribe(video_id)
        except Exception as e:
            transcript = ""
            print(f"文字起こし失敗: {e}")
        video_data.append((title, description, transcript, video_url, thumbnail))

def build_search_index():
    global index
    texts = [f"{title}. {desc}. {transcript}" for title, desc, transcript, _, _ in video_data]
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

@app.on_event("startup")
def startup_event():
    print("📺 動画・音声取得中...")
    fetch_youtube_videos()
    build_search_index()
    print(f"✅ 準備完了（動画数: {len(video_data)}）")

@app.get("/search")
def search(query: str = Query(...)):
    q_embedding = model.encode([query])
    D, I = index.search(q_embedding, k=10)
    results = []
    for idx in I[0]:
        if idx < len(video_data):
            title, desc, transcript, url, thumbnail = video_data[idx]
            results.append({
                "title": title,
                "description": desc,
                "transcript": transcript[:100] + "...",
                "url": url,
                "thumbnail": thumbnail
            })
    return results

frontend_path = pathlib.Path(__file__).parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
