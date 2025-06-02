from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
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
    log_entry = {
        "query": query,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d")
    }
    if os.path.exists(search_log_file):
        with open(search_log_file, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = []
    log.append(log_entry)
    with open(search_log_file, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
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

def highlight_text(text, query):
    if not query:
        return text
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)

@app.get("/search")
def search(query: str = Query(...)):
    log_search_query(query)
    expanded_query = expand_query(query)
    q_embedding = model.encode([expanded_query])
    D, I = index.search(q_embedding, k=10)

    results = []
    for i in I[0]:
        if i >= len(video_data):
            continue
        v = video_data[i]
        title = highlight_text(v["title"], query)
        description = highlight_text(v["description"], query)
        url = v["url"]
        video_id = url.split("v=")[-1]
        embed_url = f"https://www.youtube.com/embed/{video_id}"
        thumbnail = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        results.append({
            "title": title,
            "description": description,
            "url": url,
            "embed_url": embed_url,
            "thumbnail": thumbnail
        })
    return results

@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard(username: str = Depends(authenticate)):
    if not os.path.exists(search_log_file):
        return "<h2>まだ検索ログはありません。</h2>"

    with open(search_log_file, "r", encoding="utf-8") as f:
        logs = json.load(f)

    by_date = defaultdict(int)
    for entry in logs:
        by_date[entry["timestamp"]] += 1

    html = """
    <h2>検索ログ（🗓 日別グラフ）</h2>
    <canvas id="logChart" width="600" height="300"></canvas>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
      const data = {
        labels: %s,
        datasets: [{
          label: '検索回数',
          data: %s,
          backgroundColor: 'rgba(0, 123, 255, 0.5)'
        }]
      };
      const config = {
        type: 'bar',
        data: data,
        options: { scales: { y: { beginAtZero: true } } }
      };
      new Chart(document.getElementById('logChart'), config);
    </script>
    """ % (json.dumps(list(by_date.keys())), json.dumps(list(by_date.values())))

    html += '<br><a href="/admin/export_csv" target="_blank">📥 CSVをダウンロード</a>'
    return html

@app.get("/admin/export_csv")
def export_csv(username: str = Depends(authenticate)):
    if not os.path.exists(search_log_file):
        raise HTTPException(status_code=404, detail="ログが見つかりません")

    with open(search_log_file, "r", encoding="utf-8") as f:
        logs = json.load(f)

    by_date = defaultdict(int)
    for entry in logs:
        by_date[entry["timestamp"]] += 1

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["日付", "検索回数"])
    for date, count in sorted(by_date.items()):
        writer.writerow([date, count])

    response = StreamingResponse(iter([output.getvalue()]),
                                 media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=search_logs.csv"
    return response

frontend_path = pathlib.Path(__file__).parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
