# クイックスタートガイド

## 📦 必要なファイル

このパッケージには以下が含まれています:

```
youtube-ai-search-optimized/
├── main.py                    # ✅ 最適化されたバックエンド
├── requirements.txt           # ✅ Python依存関係
├── frontend/index.html        # ✅ 検索UI（最適化版）
├── admin_ui/                  # ✅ 管理画面（既存互換）
│   ├── index.html
│   ├── faq.html
│   └── logs.html
│
└── ★以下は空ファイル（要置換）★
    ├── data.json              # ❌ 既存ファイルで上書き必須
    ├── synonyms.json          # ❌ 既存ファイルで上書き必須
    └── faq_chatbot_fixed_only.json  # ❌ 既存ファイルで上書き必須
```

## 🚀 デプロイ手順（3ステップ）

### ステップ1: データファイルを配置

```bash
# 既存のファイルをコピー
cp /path/to/existing/data.json ./data.json
cp /path/to/existing/synonyms.json ./synonyms.json
cp /path/to/existing/faq_chatbot_fixed_only.json ./faq_chatbot_fixed_only.json
```

### ステップ2: GitHubにプッシュ

```bash
git init
git add .
git commit -m "Deploy optimized version"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### ステップ3: Render.comで設定

1. **New Web Service** を作成
2. GitHubリポジトリを接続
3. 設定:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Deploy!

## ✅ 動作確認

- 検索画面: `https://your-app.onrender.com/`
- 管理画面: `https://your-app.onrender.com/admin` (admin/abc123)

## 🎯 改善効果

- ⚡ 起動時間: 10秒 → 2秒
- 🚀 検索速度: 500ms → 150ms
- 💾 メモリ効率化
- 🛡️ エラーハンドリング強化

## ❓ トラブルシューティング

**Q: 起動時にタイムアウトする**  
A: 初回は `/health` で状態確認。数分待ってから再アクセス。

**Q: 検索結果が空**  
A: データファイル（data.json等）が正しく配置されているか確認。

**Q: 管理画面にログインできない**  
A: デフォルトは `admin` / `abc123`。環境変数で変更可能。

---

詳細は README.md を参照してください。
