from srt.ksasr import KuaiShouASR

audio_file = r"C:\Users\watermelon\Documents\WeChat Files\wxid_kg218b1bkomi21\FileStorage\File\2025-03\1.mp3"
asr = KuaiShouASR(audio_file)
asr_data = asr.run()

srt = asr_data.to_srt()
print(srt)