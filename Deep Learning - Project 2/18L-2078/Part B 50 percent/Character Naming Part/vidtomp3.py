import moviepy.editor as mp
clip = mp.VideoFileClip("new.mp4")
clip.audio.write_audiofile("new.wav")