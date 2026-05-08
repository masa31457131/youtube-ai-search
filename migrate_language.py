#!/usr/bin/env python3
"""
既存のdata.jsonとfaq.jsonに言語タグを一括追加するスクリプト

使用方法:
    python migrate_language.py

注意:
    - 既存データのバックアップを作成します
    - デフォルトで全データに "language": "ja" を追加します
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

def migrate_videos():
    """data.jsonに言語タグを追加"""
    data_path = Path("data.json")
    
    if not data_path.exists():
        print("❌ data.json が見つかりません")
        return
    
    # バックアップを作成
    backup_path = Path(f"data.json.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy(data_path, backup_path)
    print(f"✅ バックアップを作成しました: {backup_path}")
    
    # データを読み込み
    with open(data_path, 'r', encoding='utf-8') as f:
        videos = json.load(f)
    
    # 言語タグを追加
    modified = 0
    for video in videos:
        if 'language' not in video:
            video['language'] = 'ja'  # デフォルトは日本語
            modified += 1
    
    # 保存
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(videos, f, ensure_ascii=False, indent=2)
    
    print(f"✅ data.json: {modified} 件に言語タグを追加しました（全{len(videos)}件）")

def migrate_faq():
    """faq.jsonに言語タグを追加"""
    faq_path = Path("faq.json")
    
    if not faq_path.exists():
        print("❌ faq.json が見つかりません")
        return
    
    # バックアップを作成
    backup_path = Path(f"faq.json.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy(faq_path, backup_path)
    print(f"✅ バックアップを作成しました: {backup_path}")
    
    # データを読み込み
    with open(faq_path, 'r', encoding='utf-8') as f:
        faq_data = json.load(f)
    
    # 言語タグを追加
    modified = 0
    if 'faqs' in faq_data:
        for faq in faq_data['faqs']:
            if 'language' not in faq:
                faq['language'] = 'ja'  # デフォルトは日本語
                modified += 1
    
    # 保存
    with open(faq_path, 'w', encoding='utf-8') as f:
        json.dump(faq_data, f, ensure_ascii=False, indent=2)
    
    total = len(faq_data.get('faqs', []))
    print(f"✅ faq.json: {modified} 件に言語タグを追加しました（全{total}件）")

def main():
    print("=" * 60)
    print("言語タグ移行スクリプト")
    print("=" * 60)
    print()
    
    # 確認
    response = input("既存データに 'language': 'ja' を追加します。続行しますか？ (y/N): ")
    if response.lower() != 'y':
        print("❌ キャンセルしました")
        return
    
    print()
    
    # 移行実行
    migrate_videos()
    migrate_faq()
    
    print()
    print("=" * 60)
    print("✅ 移行が完了しました")
    print("=" * 60)
    print()
    print("次のステップ:")
    print("1. サーバーを再起動してください")
    print("2. 管理画面で動画・FAQの言語タグを個別に編集できます")
    print("3. 英語版のデータを追加する場合は 'language': 'en' を設定してください")
    print()

if __name__ == "__main__":
    main()
