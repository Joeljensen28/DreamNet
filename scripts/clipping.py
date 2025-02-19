from pydub import AudioSegment
import os

def clip_song(song_path, output_dir, segment_len=1000):
    song = AudioSegment.from_wav(song_path)
    for i in range(0, len(song), segment_len):
        pieces = song_path.split('/')
        name = pieces[len(pieces)-1]
        clip = song[i:i+segment_len]
        clip.export(os.path.join(output_dir, f'clip_{i // segment_len}_{name}'), format='wav')

for file in os.listdir('/Users/joeljensen28/Documents/Personal/programming/DreamNet/data'):
    if file.endswith('.wav'):
        clip_song(file, '/Users/joeljensen28/Documents/Personal/programming/DreamNet/data/train_clips')