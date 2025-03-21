from gtts import gTTS
import os

def text_to_speech(text, filename="output.mp3"):
    """
    Converts the given text into Hindi speech and saves it as an audio file.
    """
    tts = gTTS(text=text, lang="hi")  # Convert text to Hindi speech
    tts.save(filename)  # Save as MP3 file
    return filename

# Example Usage:
if __name__ == "__main__":
    summary_text = "टेस्ला की खबरें ज्यादातर सकारात्मक हैं। संभावित स्टॉक वृद्धि की उम्मीद है।"
    audio_file = text_to_speech(summary_text, "summary.mp3")
    
    # Play the audio file (Windows)
    os.system(f"start {audio_file}")  
    print(f"Hindi speech saved as {audio_file}")
