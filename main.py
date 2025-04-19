
from fastapi import FastAPI, Query, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
import pathlib
from collections import Counter
import csv
import io
import secrets

app = FastAPI()
security = HTTPBasic()

USERNAME = "admin"
PASSWORD = "pass123"  # 必要に応じて変更可能

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
video_data = []
index = None
search_log_file = "search_logs.json"

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
    load_data()
    build_search_index()

def log_search_query(query: str):
    try:
        if os.path.exists(search_log_file):
            with open(search_log_file, "r", encoding="utf-8") as f:
                log = json.load(f)
        else:
            log = []
        log.append(query)
        with open(search_log_file, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] 検索ログ保存に失敗: {e}")

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="認証に失敗しました", headers={"WWW-Authenticate": "Basic"})
    return credentials.username

@app.get("/search")
def search(query: str = Query(...)):
    log_search_query(query)
    q_embedding = model.encode([query])
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

frontend_path = pathlib.Path(__file__).parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
