from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
import pathlib
import io
import csv
import secrets
from collections import Counter, defaultdict
import re
from datetime import datetime

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

model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")

video_data = []
text_corpus = []
index = None

search_log_file = "search_logs.json"

def clean_text(text):
    return re.sub(r"[ \n\r\t]+", " ", text).strip()

def load_data():
    global video_data, text_corpus
    with open("data.json", "r", encoding="utf-8") as f:
        video_data = json.load(f)
    text_corpus = [
        clean_text(f"{v['title']}。{v['description']}。{v.get('transcript', '')}")
        for v in video_data
    ]

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
    log = []
    if os.path.exists(search_log_file):
        with open(search_log_file, "r", encoding="utf-8") as f:
            log = json.load(f)
    log.append({"query": query, "date": datetime.now().strftime("%Y-%m-%d")})
    with open(search_log_file, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if not (secrets.compare_digest(credentials.username, USERNAME) and
            secrets.compare_digest(credentials.password, PASSWORD)):
        raise HTTPException(status_code=401, detail="認証に失敗しました", headers={"WWW-Authenticate": "Basic"})
    return credentials.username

def expand_query(query: str) -> str:
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
    return query + " " + " ".join(expansion)

@app.get("/search")
def search(query: str = Query(...)):
    log_search_query(query)
    expanded_query = expand_query(query)
    q_embedding = model.encode([expanded_query], convert_to_numpy=True)
    D, I = index.search(q_embedding, k=10)

    results = []
    for idx in I[0]:
        if idx < len(video_data):
            v = video_data[idx]
            highlighted_title = re.sub(f"({re.escape(query)})", r"<mark>\1</mark>", v["title"], flags=re.IGNORECASE)
            highlighted_description = re.sub(f"({re.escape(query)})", r"<mark>\1</mark>", v["description"], flags=re.IGNORECASE)
            results.append({
                "title": highlighted_title,
                "description": highlighted_description,
                "url": v["url"],
                "thumbnail": v.get("thumbnail", f"https://i.ytimg.com/vi/{v['url'].split('v=')[-1]}/mqdefault.jpg")
            })
    return results

@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard(username: str = Depends(authenticate)):
    if not os.path.exists(search_log_file):
        return "<h2>まだ検索ログはありません。</h2>"

    with open(search_log_file, "r", encoding="utf-8") as f:
        logs = json.load(f)

    counts = Counter([log["query"] for log in logs]).most_common()
    daily_counts = defaultdict(int)
    for log in logs:
        daily_counts[log["date"]] += 1

    html = "<h2>検索ログ集計</h2><ul>"
    for word, count in counts:
        html += f"<li><strong>{word}</strong>: {count} 回</li>"
    html += "</ul>"

    html += "<h3>📅 日別検索件数</h3><ul>"
    for day, count in sorted(daily_counts.items()):
        html += f"<li>{day}: {count} 件</li>"
    html += "</ul>"

    html += '<a href="/admin/export_csv" target="_blank">📥 CSVをダウンロード</a>'
    return html

@app.get("/admin/export_csv")
def export_csv(username: str = Depends(authenticate)):
    if not os.path.exists(search_log_file):
        raise HTTPException(status_code=404, detail="ログが見つかりません")

    with open(search_log_file, "r", encoding="utf-8") as f:
        logs = json.load(f)

    counts = Counter([log["query"] for log in logs]).most_common()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["検索キーワード", "回数"])
    for word, count in counts:
        writer.writerow([word, count])

    response = StreamingResponse(iter([output.getvalue()]),
                                 media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=search_logs.csv"
    return response

frontend_path = pathlib.Path(__file__).parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
