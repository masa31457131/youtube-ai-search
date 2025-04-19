
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
from fastapi.staticfiles import StaticFiles
import pathlib

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

def load_data():
    global video_data
    with open("data.json", "r", encoding="utf-8") as f:
        video_data.extend(json.load(f))

def build_search_index():
    global index
    texts = [f"{v['title']}. {v['description']}. {v['transcript']}" for v in video_data]
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

@app.on_event("startup")
def startup_event():
    print("📁 data.json 読み込み中...")
    load_data()
    build_search_index()
    print(f"✅ {len(video_data)}件ロード完了")

@app.get("/search")
def search(query: str = Query(...)):
    q_embedding = model.encode([query])
    D, I = index.search(q_embedding, k=10)
    results = [video_data[i] for i in I[0] if i < len(video_data)]
    return results

frontend_path = pathlib.Path(__file__).parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
