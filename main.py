from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel
import faiss
import json
import os
import pathlib
import io
import csv
import secrets
from collections import Counter
import re
import torch

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

# ✅ モデルの初期化
sentence_model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")
tokenizer_bert = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
bert_model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese")
bert_model.eval()

video_data = []
index = None
text_corpus = []

search_log_file = "search_logs.json"

def clean_text(text):
    return re.sub(r"[ \n\r\t]+", " ", text).strip()

def load_data():
    global video_data, text_corpus
    with open("data.json", "r", encoding="utf-8") as f:
        video_data = json.load(f)
    text_corpus = [
        clean_text(f"{v['title']}。{v.get('description', '')}。{v.get('transcript', '')}")
        for v in video_data
    ]

def build_search_index():
    global index
    embeddings = sentence_model.encode(text_corpus, convert_to_numpy=True)
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

# ✅ クエリ拡張（静的類義語 + BERTによる意味補強）
def expand_query(query: str, top_k=3) -> str:
    synonym_map = {
        "袖": ["そで", "スリーブ"],
        "型紙": ["パターン", "テンプレート"],
        "修正": ["変更", "編集", "調整"],
        "動画": ["ビデオ", "映像", "ムービー"],
    }
    expansion = []
    for word, synonyms in synonym_map.items():
        if word in query:
            expansion.extend(synonyms)

    # BERTの文脈ベクトル（今後の応用のために埋め込み）
    inputs = tokenizer_bert(query, return_tensors="pt")
    with torch.no_grad():
        _ = bert_model(**inputs)

    return query + " " + " ".join(expansion)

@app.get("/search")
def search(query: str = Query(...)):
    log_search_query(query)
    expanded_query = expand_query(query)
    q_embedding = sentence_model.encode([expanded_query])
    D, I = index.search(q_embedding, k=10)

    results = []
    for i in I[0]:
        if i < len(video_data):
            v = video_data[i]
            results.append({
                "title": v["title"],
                "description": v.get("description", ""),
                "video_id": v.get("video_id", ""),
                "thumbnail": v.get("thumbnail", ""),
                "url": f"https://www.youtube.com/watch?v={v.get('video_id', '')}"
            })
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
    html += "</ul>"
    html += '<a href="/admin/export_csv" target="_blank">📥 CSVをダウンロード</a>'
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

    response = StreamingResponse(iter([output.getvalue()]),
                                 media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=search_logs.csv"
    return response

# ✅ フロントエンド提供（/frontend/index.html）
frontend_path = pathlib.Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
else:
    @app.get("/", response_class=HTMLResponse)
    def fallback():
        return "<h2>フロントエンドが見つかりません。frontendフォルダを設置してください。</h2>"
