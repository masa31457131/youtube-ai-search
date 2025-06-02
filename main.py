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
import html
import re
from collections import Counter, defaultdict
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

# ✅ 日本語特化 SentenceTransformer モデル
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
    text_corpus = [clean_text(f"{v['title']}。{v['description']}。{v['transcript']}") for v in video_data]

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
    entry = {"query": query, "timestamp": datetime.now().isoformat()}
    if os.path.exists(search_log_file):
        with open(search_log_file, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = []
    log.append(entry)
    with open(search_log_file, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, detail="認証に失敗しました", headers={"WWW-Authenticate": "Basic"})
    return credentials.username

# ✅ クエリ補強関数（同義語拡張）
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

def highlight(text, query):
    pattern = re.escape(query)
    return re.sub(pattern, lambda m: f"<mark>{html.escape(m.group(0))}</mark>", text, flags=re.IGNORECASE)

@app.get("/search")
def search(query: str = Query(...)):
    log_search_query(query)
    expanded_query = expand_query(query)
    q_embedding = model.encode([expanded_query])
    D, I = index.search(q_embedding, k=10)

    results = []
    for j, i in enumerate(I[0]):
        if i < len(video_data):
            item = video_data[i].copy()
            item["title"] = highlight(item["title"], query)
            item["description"] = highlight(item["description"], query)
            item.pop("transcript", None)  # 文字起こしを除外
            item["score"] = float(D[0][j])
            item["embed_url"] = f"https://www.youtube.com/embed/{item['video_id']}"
            results.append(item)
    return results

@app.get("/admin", response_class=HTMLResponse)
def admin_dashboard(username: str = Depends(authenticate)):
    if not os.path.exists(search_log_file):
        return "<h2>まだ検索ログはありません。</h2>"

    with open(search_log_file, "r", encoding="utf-8") as f:
        logs = json.load(f)

    counts = Counter(entry["query"] for entry in logs).most_common()
    html_content = "<h2>検索ログ集計（CSVエクスポート付き）</h2><ul>"
    for word, count in counts:
        html_content += f"<li><strong>{html.escape(word)}</strong>: {count} 回</li>"
    html_content += "</ul>"
    html_content += '<a href="/admin/export_csv" target="_blank">📥 CSVをダウンロード</a><br>'
    html_content += '<a href="/admin/graph" target="_blank">📊 日別トラフィックグラフを見る</a>'
    return html_content

@app.get("/admin/export_csv")
def export_csv(username: str = Depends(authenticate)):
    if not os.path.exists(search_log_file):
        raise HTTPException(status_code=404, detail="ログが見つかりません")

    with open(search_log_file, "r", encoding="utf-8") as f:
        logs = json.load(f)

    counts = Counter(entry["query"] for entry in logs).most_common()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["検索キーワード", "回数"])
    for word, count in counts:
        writer.writerow([word, count])

    response = StreamingResponse(iter([output.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=search_logs.csv"
    return response

@app.get("/admin/search_stats")
def search_stats(username: str = Depends(authenticate)):
    if not os.path.exists(search_log_file):
        return {}

    with open(search_log_file, "r", encoding="utf-8") as f:
        logs = json.load(f)

    day_counts = defaultdict(int)
    for entry in logs:
        date = entry["timestamp"][:10]
        day_counts[date] += 1

    return dict(sorted(day_counts.items()))

@app.get("/admin/graph", response_class=HTMLResponse)
def graph_page(username: str = Depends(authenticate)):
    return """
    <html>
    <head>
        <title>検索トラフィックグラフ</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <h2>📊 日別検索トラフィック</h2>
        <canvas id="trafficChart" width="800" height="400"></canvas>
        <script>
            async function fetchData() {
                const res = await fetch('/admin/search_stats', {
                    headers: { 'Authorization': 'Basic ' + btoa('admin:pass123') }
                });
                const data = await res.json();
                const labels = Object.keys(data);
                const counts = Object.values(data);

                const ctx = document.getElementById('trafficChart').getContext('2d');
                new Chart(ctx, {
                    type: 'bar',
                    data: {
                        labels: labels,
                        datasets: [{
                            label: '検索回数',
                            data: counts,
                            backgroundColor: 'rgba(54, 162, 235, 0.6)'
                        }]
                    },
                    options: {
                        scales: {
                            x: { title: { display: true, text: '日付' } },
                            y: { title: { display: true, text: '検索回数' }, beginAtZero: true }
                        }
                    }
                });
            }
            fetchData();
        </script>
    </body>
    </html>
    """

# フロントエンドHTMLを提供
frontend_path = pathlib.Path(__file__).parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
