from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sentence_transformers import SentenceTransformer, util
import torch
import faiss
import json
import os
import pathlib
import io
import csv
import secrets
from collections import Counter
import re

app = FastAPI()
security = HTTPBasic()

USERNAME = "admin"
PASSWORD = "pass123"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 日本語特化 Sentence-BERT モデル
model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")

video_data = []
index = None
text_corpus = []

search_log_file = "search_logs.json"

# ✅ 類義語辞書の読み込み
def load_synonyms():
    try:
        with open("synonyms.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

synonym_map = load_synonyms()

def clean_text(text):
    return re.sub(r"[ \n\r\t]+", " ", text).strip()

def load_data():
    global video_data, text_corpus
    with open("data.json", "r", encoding="utf-8") as f:
        video_data = json.load(f)

    for v in video_data:
        if not v.get("thumbnail") and v.get("video_id"):
            v["thumbnail"] = f"https://img.youtube.com/vi/{v['video_id']}/mqdefault.jpg"

    text_corpus.clear()
    for v in video_data:
        combined = clean_text(f"{v['title']}。{v.get('description', '')}。{v.get('transcript', '')}")
        text_corpus.append(combined)

def build_search_index():
    global index
    embeddings = model.encode(text_corpus, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

@app.on_event("startup")
def startup_event():
    load_data()
    build_search_index()

def log_search_query(query: str):
    if os.path.exists(search_log_file):
        with open(search_log_file, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = []
    log.append(query)
    with open(search_log_file, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="認証に失敗しました", headers={"WWW-Authenticate": "Basic"})
    return credentials.username

# ✅ 類義語展開（JSON + BERT自動）
def expand_query(query: str) -> str:
    expansion = []
    for word, synonyms in synonym_map.items():
        if word in query:
            expansion.extend(synonyms)

    # BERTベースの類似語を自動で追加
    candidate_words = list(set(" ".join(text_corpus).split()))
    candidate_embeddings = model.encode(candidate_words, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
    top_results = torch.topk(cos_scores, k=3)

    for idx in top_results.indices:
        similar_word = candidate_words[idx]
        if similar_word not in query:
            expansion.append(similar_word)

    return query + " " + " ".join(expansion)

@app.get("/search")
def search(query: str = Query(...)):
    log_search_query(query)
    expanded_query = expand_query(query)
    q_embedding = model.encode([expanded_query])
    D, I = index.search(q_embedding, k=10)
    results = [video_data[i] for i in I[0] if i < len(video_data)]
    return results

@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard(username: str = Depends(authenticate)):
    if not os.path.exists(search_log_file):
        return "<h2>まだ検索ログはありません。</h2>"

    with open(search_log_file, "r", encoding="utf-8") as f:
        logs = json.load(f)

    counts = Counter(logs).most_common()
    html = "<h2>検索ログ集計（CSVエクスポート付き）</h2><ul>"
    for word, count in counts:
        html += f"<li><strong>{word}</strong>: {count} 回</li>"
    html += "</ul><a href='/admin/export_csv' target='_blank'>📥 CSVをダウンロード</a>"
    return html

@app.get("/admin/export_csv")
def export_csv(username: str = Depends(authenticate)):
    if not os.path.exists(search_log_file):
        raise HTTPException(status_code=404, detail="ログが見つかりません")

    with open(search_log_file, "r", encoding="utf-8") as f:
        logs = json.load(f)

    counts = Counter(logs).most_common()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["検索キーワード", "回数"])
    for word, count in counts:
        writer.writerow([word, count])

    response = StreamingResponse(iter([output.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=search_logs.csv"
    return response

# ✅ フロントエンド表示
frontend_path = pathlib.Path(__file__).parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_file = frontend_path / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html が見つかりません")
    return index_file.read_text(encoding="utf-8")
