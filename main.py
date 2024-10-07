import numpy as np
from IPython.display import Audio
from scipy.io.wavfile import write
from models.models import WakeupModel,\
                        TextToImageModel,\
                        Text2TextGenerationModel,\
                        SysthesiseSpeechModel,\
                        SpeechTranscriptionModel
    

def main():
    print("WAKE UP MODEL...")
    WakeupModel().launch_fn()
    print("SPEECH TRANSCRIPTION MODEL...")
    transcription = SpeechTranscriptionModel(model_called="openai/whisper-base.en").transcribe()
    print("LANGUAGE MODEL QUERY.....\n")
    response = Text2TextGenerationModel().generate(transcription)
    response = f"{transcription} {response.replace('<pad>','').replace('</s>','')}"
    print(f"SYSTHESISE SPEECH MODEL....")
    audio = SysthesiseSpeechModel().systhesise(text = response)
    audio = np.int16(audio.detach().cpu().numpy()/max(abs(audio))*32767)
    write("test.wav", 16000, audio)
    # print("TEXT TO IMAGE MODEL...")
    # TextToImageModel().image(prompt = transcription,)
    print("DONE...")


if __name__ == "__main__":
    main()