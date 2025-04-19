
import whisper
from pytube import YouTube
import json
import os
import requests

API_KEY = "YOUR_API_KEY"
CHANNEL_ID = "YOUR_CHANNEL_ID"
MAX_VIDEOS = 5

whisper_model = whisper.load_model("base")

def download_and_transcribe(video_id):
    yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
    audio_stream = yt.streams.filter(only_audio=True).first()
    filename = f"audio_{video_id}.mp4"
    audio_stream.download(filename=filename)
    result = whisper_model.transcribe(filename, language="ja")
    os.remove(filename)
    return result["text"]

def fetch_video_data():
    url = f"https://www.googleapis.com/youtube/v3/search?key={API_KEY}&channelId={CHANNEL_ID}&part=snippet&type=video&maxResults={MAX_VIDEOS}"
    res = requests.get(url)
    data = res.json()
    return data.get("items", [])

def main():
    items = fetch_video_data()
    output = []
    for item in items:
        snippet = item["snippet"]
        title = snippet["title"]
        description = snippet["description"]
        video_id = item["id"]["videoId"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        thumbnail = snippet["thumbnails"]["medium"]["url"]
        print(f"▶️ {title}")
        try:
            transcript = download_and_transcribe(video_id)
        except Exception as e:
            print(f"❌ 失敗: {e}")
            transcript = ""
        output.append({
            "title": title,
            "description": description,
            "transcript": transcript,
            "url": url,
            "thumbnail": thumbnail
        })

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
