import argparse
import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI


def find_audio_files(path: str) -> List[str]:
    supported_exts = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".ogg", ".webm", ".aac", ".flac"}
    if os.path.isdir(path):
        files: List[str] = []
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if os.path.isfile(full) and os.path.splitext(name)[1].lower() in supported_exts:
                files.append(full)
        return sorted(files)
    else:
        ext = os.path.splitext(path)[1].lower()
        return [path] if ext in supported_exts and os.path.isfile(path) else []


def transcribe_file(client: OpenAI, audio_path: str, force_english: bool = True, language: Optional[str] = None) -> str:
    # Use translations endpoint to force English; otherwise regular transcription
    with open(audio_path, "rb") as f:
        if force_english:
            result = client.audio.translations.create(
                model="whisper-1",
                file=f,
                response_format="json",
                temperature=0,
            )
            return getattr(result, "text", "").strip()
        else:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="json",
                temperature=0,
                language=language if language else None,
            )
            return getattr(result, "text", "").strip()


def save_transcript(text: str, audio_path: str, out_dir: Optional[str]) -> str:
    base = os.path.splitext(os.path.basename(audio_path))[0]
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{base}.txt")
    else:
        out_path = os.path.join(os.path.dirname(audio_path), f"{base}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio to English .txt using OpenAI.")
    parser.add_argument("input", help="Path to an audio file or a directory containing audio files.")
    parser.add_argument("--out-dir", dest="out_dir", default=None, help="Optional output directory for .txt files.")
    parser.add_argument("--no-english", action="store_true", help="Do not force translation to English.")
    parser.add_argument("--language", dest="language", default=None, help="Hint source language code (e.g., it for Italian).")
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI()

    files = find_audio_files(args.input)
    if not files:
        print("No audio files found. Supported: .mp3 .wav .m4a .mp4 .mpeg .ogg .webm .aac .flac")
        return

    print(f"Found {len(files)} audio file(s). Starting transcription...")

    for idx, audio_path in enumerate(files, start=1):
        try:
            print(f"[{idx}/{len(files)}] Transcribing: {audio_path}")
            text = transcribe_file(
                client,
                audio_path,
                force_english=(not args.no_english),
                language=args.language,
            )
            output_path = save_transcript(text, audio_path, args.out_dir)
            print(f"Saved: {output_path}")
        except Exception as e:
            print(f"Failed to transcribe {audio_path}: {e}")

    print("All done.")


if __name__ == "__main__":
    main()



