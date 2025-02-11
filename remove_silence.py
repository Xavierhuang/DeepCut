from pydub import AudioSegment, silence

audio = AudioSegment.from_file("sweetheartrecap.MP3", format="mp3")

chunks = silence.split_on_silence(
    audio,
    min_silence_len=300,        # 0.5s
    silence_thresh=-50,         # or audio.dBFS - 16, etc.
    keep_silence=100            # keep 100 ms around each chunk
)

processed = AudioSegment.empty()
for chunk in chunks:
    processed += chunk

processed.export("output.mp3", format="mp3")

