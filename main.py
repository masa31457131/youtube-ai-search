from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import uvicorn
import os

app = FastAPI()

# CORS設定（必要に応じてドメイン制限可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルとデータの読み込み
model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")

with open("data/video_data.json", "r", encoding="utf-8") as f:
    video_data = json.load(f)

# ✅ タイトル + 説明 + 文字起こし を全文ベクトル化
texts = [
    f"タイトル: {v['title']} 説明: {v.get('description', '')} 本文: {v.get('transcript', '')}"
    for v in video_data
]
embeddings = model.encode(texts, convert_to_numpy=True)

# faissに埋め込みを登録
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

@app.get("/search")
async def search(request: Request):
    query = request.query_params.get("q", "")
    if not query:
        return JSONResponse(content={"error": "検索語が指定されていません"}, status_code=400)

    # ✅ クエリに意味補強を追加
    enhanced_query = f"この文章の意味に関連する内容を検索: {query}"
    q_embedding = model.encode([enhanced_query], convert_to_numpy=True)

    k = 10  # 上位10件を返す
    D, I = index.search(q_embedding, k)

    results = []
    for i, idx in enumerate(I[0]):
        if idx < len(video_data):
            video = video_data[idx]
            results.append({
                "title": video["title"],
                "video_id": video["video_id"],
                "url": f"https://www.youtube.com/watch?v={video['video_id']}",
                "transcript": video.get("transcript", "")[:300] + "...",  # 抜粋表示
            })

    return {"results": results}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
