// フロントの状態を簡単に管理するだけのスクリプト

function getBackendBaseUrl() {
  const input = document.getElementById("backend-url");
  return input.value.trim().replace(/\/+$/, ""); // 末尾の / を削る
}

async function registerVideo() {
  const backend = getBackendBaseUrl();
  const statusEl = document.getElementById("register-video-status");
  statusEl.textContent = "";

  if (!backend) {
    statusEl.textContent = "Backend URL を先に入力してください。";
    return;
  }

  const url = document.getElementById("video-url").value.trim();
  const language = document.getElementById("video-language").value || null;

  if (!url) {
    statusEl.textContent = "YouTube URL を入力してください。";
    return;
  }

  try {
    statusEl.textContent = "動画を登録中...(Whisperで文字起こしするので少し時間がかかります)";
    const res = await fetch(`${backend}/videos/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url, language }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }

    const data = await res.json();
    statusEl.textContent = `登録完了: [${data.id}] ${data.title} (status=${data.status})`;
  } catch (err) {
    console.error(err);
    statusEl.textContent = "エラー: " + err.message;
  }
}

async function search() {
  const backend = getBackendBaseUrl();
  const statusEl = document.getElementById("search-status");
  const resultsEl = document.getElementById("search-results");
  statusEl.textContent = "";
  resultsEl.innerHTML = "";

  if (!backend) {
    statusEl.textContent = "Backend URL を先に入力してください。";
    return;
  }

  const query = document.getElementById("search-query").value.trim();
  const topK = parseInt(document.getElementById("search-topk").value, 10) || 5;

  if (!query) {
    statusEl.textContent = "検索クエリを入力してください。";
    return;
  }

  try {
    statusEl.textContent = "検索中...";
    const res = await fetch(`${backend}/search/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }

    const data = await res.json();
    statusEl.textContent = `検索完了 (${data.hits.length}件 / ${data.latency_ms ?? "?"}ms)`;

    if (!data.hits.length) {
      resultsEl.textContent = "該当するセグメントはありませんでした。";
      return;
    }

    data.hits.forEach((hit) => {
      const item = document.createElement("div");
      item.className = "result-item";

      const title = document.createElement("div");
      title.className = "result-title";

      // YouTube の再生位置リンク
      const url = new URL(hit.url);
      url.searchParams.set("t", `${hit.start_sec}s`);

      title.innerHTML = `<a href="${url.toString()}" target="_blank" rel="noopener noreferrer">
        [${hit.video_id}] ${hit.video_title}
      </a>`;

      const text = document.createElement("div");
      text.className = "result-text";
      text.textContent = hit.text;

      const meta = document.createElement("div");
      meta.className = "result-meta";
      meta.textContent = `開始: ${hit.start_sec}s / 終了: ${hit.end_sec}s / score: ${hit.score.toFixed(
        3
      )}`;

      item.appendChild(title);
      item.appendChild(text);
      item.appendChild(meta);
      resultsEl.appendChild(item);
    });
  } catch (err) {
    console.error(err);
    statusEl.textContent = "エラー: " + err.message;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document
    .getElementById("register-video-btn")
    .addEventListener("click", registerVideo);
  document.getElementById("search-btn").addEventListener("click", search);
});
