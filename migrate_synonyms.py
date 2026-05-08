#!/usr/bin/env python3
"""
synonyms.jsonを言語別構造に変換するスクリプト

旧形式:
{
  "プロッタ": ["プロッター", "大判プリンタ"],
  "パスワード": ["PW", "pass"]
}

新形式:
{
  "ja": {
    "プロッタ": ["プロッター", "大判プリンタ"],
    "パスワード": ["PW", "pass"]
  },
  "en": {}
}
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

def migrate_synonyms():
    """synonyms.jsonを言語別構造に変換"""
    synonyms_path = Path("synonyms.json")
    
    if not synonyms_path.exists():
        print("❌ synonyms.json が見つかりません")
        return
    
    # バックアップを作成
    backup_path = Path(f"synonyms.json.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy(synonyms_path, backup_path)
    print(f"✅ バックアップを作成しました: {backup_path}")
    
    # データを読み込み
    with open(synonyms_path, 'r', encoding='utf-8') as f:
        old_data = json.load(f)
    
    # 既に新形式かチェック
    if 'ja' in old_data or 'en' in old_data:
        print("✅ 既に新形式です。変換は不要です。")
        return
    
    # 新形式に変換
    new_data = {
        "ja": old_data,  # 既存データは日本語として扱う
        "en": {}          # 英語は空
    }
    
    # 保存
    with open(synonyms_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 変換完了: {len(old_data)} 件の同義語を日本語（ja）に移動しました")
    print(f"   英語（en）は空で作成されました")

def main():
    print("=" * 60)
    print("synonyms.json 言語別構造変換スクリプト")
    print("=" * 60)
    print()
    
    # 確認
    response = input("synonyms.jsonを言語別構造に変換します。続行しますか？ (y/N): ")
    if response.lower() != 'y':
        print("❌ キャンセルしました")
        return
    
    print()
    
    # 変換実行
    migrate_synonyms()
    
    print()
    print("=" * 60)
    print("✅ 変換が完了しました")
    print("=" * 60)
    print()
    print("次のステップ:")
    print("1. サーバーを再起動してください")
    print("2. 管理画面の「同義語編集」で言語別に同義語を編集できます")
    print("3. 英語の同義語を追加する場合は、Englishタブで追加してください")
    print()

if __name__ == "__main__":
    main()
