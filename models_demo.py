import torch
from transformers import pipeline
from datasets import load_dataset
from tools.tools import device, translate
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from transformers.pipelines.audio_utils import ffmpeg_microphone_live

DEVICE = device()


class SpeechTranslationModel:
    
    def __init__(self, 
                 task_model: str = "automatic-speech-recognition",
                 model_called: str = "openai/whisper-base"):
        self.task_model = task_model
        self.model_called = model_called
        
    @property
    def pipe(self):
        return pipeline(self.task_model,
                        model=self.model_called,
                        device=DEVICE)
        
    def translate_speech_to_text(self, audio):
        result = translate(audio=audio,
                           pipe=self.pipe,
                           max_new_tokens=256,
                           generate_kwargs={"task": "translate"})
        return result 
    



class TextToSpeechModel:
    
    def __init__(self, 
                 model_tts_called: str = "microsoft/speecht5_tts",
                 model_vocoder_called: str = "microsoft/speecht5_hifigan",
                 embeddings_dts_path: str = "Matthijs/cmu-arctic-xvectors"):
        self.model_tts = model_tts_called
        self.model_vocoder = model_vocoder_called
        self.embeddings_dts = embeddings_dts_path
        
    @property
    def processor(self):
          return SpeechT5Processor.from_pretrained(self.model_tts)
      
    @property
    def model(self):
        model_tts = SpeechT5ForTextToSpeech.from_pretrained(self.model_tts)
        model_tts.to(DEVICE)
        return model_tts
    
    @property
    def vocoder(self):
        model_vcd = SpeechT5HifiGan.from_pretrained(self.model_vocoder)
        model_vcd.to(DEVICE)
        
    @property
    def speaker_embeddings(self):
        embeddings_dataset = load_dataset(self.embeddings_dts, split="validation")
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        return speaker_embedding
    
    def systhesise(self, text: str):
        inputs = self.processor(text = text, return_tensors = "pt")
        speech = self.model.generate_speech(
            inputs["input_ids"].to(DEVICE),
            self.speaker_embeddings.to(DEVICE),
            
        )
        return speech.cpu()
    
    
class WakeupModel:
    def __init__(self,
                 task: str = "audio-classification",
                 model_called: str = "MTT/ast-finetuned-speech-commands-v2"):
        self.task = task
        self.model_called = model_called
        
    
    @property
    def pipe(self):
        return pipeline(self.task,
                        model = self.model_called,
                        device=DEVICE)
        
    def launch_fn(self,
                  wake_word: str = "marvin",
                  prob_threhold: float = 0.5,
                  chunk_length_s: float = 2.0,
                  stream_chunk_s: float = 0.25,
                  debug: bool = False,
                  ):
        if wake_word not in self.pipe.model.config.label2id.keys():
            raise ValueError(
                f"Wake word {wake_word} not in set of valid class labels, pick a wake word in the set {self.pipe.model.config.label2id.keys()}"
            )
        
        sampling_rate = self.pipe.feature_extractor.sampling_rate
        
        mic = ffmpeg_microphone_live(
            sampling_rate=sampling_rate,
            chunk_length_s=chunk_length_s,
            stream_chunk_s=stream_chunk_s
        )
        
        print("Listening for wake word.....")
        for prediction in self.pipe(mic):
            prediction = prediction[0]
            if debug:
                print(prediction)
            if prediction["label"] == wake_word:
                if prediction["score"] > prob_threhold:
                    return True
        
        

    