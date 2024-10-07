import numpy as np
import gradio as gr
from IPython.display import Audio

from models.models import SpeechTranslationModel, TextToSpeechModel

def speech_to_speech_translation(audio):
    target_dtype = np.int16
    max_range = np.iinfo(target_dtype).max
    translated_text = SpeechTranslationModel().translate_speech_to_text(audio=audio)
    systhesised_speech = TextToSpeechModel().systhesise(text=translated_text)
    systhesised_speech = (systhesised_speech.numpy()*max_range).astype(target_dtype)
    return 16000, systhesised_speech


def demo():
    demo = gr.Blocks()
    
    mic_translate = gr.Interface(
        fn = speech_to_speech_translation,
        inputs=gr.Audio(sources="microphone", type="filepath"),
        outputs=gr.Audio(label = "Generated Speech", type="numpy"),
        
    )
    
    file_translate = gr.Interface(
    fn=speech_to_speech_translation,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=gr.Audio(label="Generated Speech", type="numpy"),
    )

    with demo:
        gr.TabbedInterface([mic_translate, file_translate], ["Microphone", "Audio File"])

    demo.launch(debug=True, share=True)