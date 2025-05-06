from pydub import AudioSegment
from pydub.playback import play

# Set the paths for both ffmpeg and ffprobe
AudioSegment.ffmpeg = "C:/ffmpeg-2025-05-05-git-f4e72eb5a3-full_build/bin/ffmpeg.exe"
AudioSegment.ffprobe = "C:/ffmpeg-2025-05-05-git-f4e72eb5a3-full_build/bin/ffprobe.exe"

# Load the audio file
audio = AudioSegment.from_file("C:/Users/hp/Documents/GitHub/Hand-gesture-music-control/rock.mp3")

# Play the audio
play(audio)
