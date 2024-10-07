
import sys
import torch
import warnings
warnings.filterwarnings("ignore")
#warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
from tools.tools import device

from diffusers import FluxPipeline
from datasets import load_dataset
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.pipelines.audio_utils import ffmpeg_microphone_live
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

DEVICE = device()


class WakeupModel:
    def __init__(self,
                 task: str = "audio-classification",
                 model_called: str = "MIT/ast-finetuned-speech-commands-v2"):
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
                
                

class SpeechTranscriptionModel:
    def __init__(self,
                 task: str = "automatic-speech-recognition",
                 model_called: str = "openai/whisper-base.en"):
        self.task = task
        self.model_called = model_called
        
    @property
    def pipe(self):
        return pipeline(self.task,
                        model = self.model_called,
                        device=DEVICE)
    
    def transcribe(self,
                   chunk_length_s: float = 10.0,
                   stream_chunk_s: float = 5.0):
        sampling_rate = self.pipe.feature_extractor.sampling_rate
        
        mic = ffmpeg_microphone_live(
            sampling_rate=sampling_rate,
            chunk_length_s=chunk_length_s,
            stream_chunk_s=stream_chunk_s
        )  
        
        print(f"Start speaking .......")
        for item in self.pipe(mic, generate_kwargs = {"max_new_tokens": 256}):
            sys.stdout.write("\003[K") 
            print(item["text"], end="\r")
            if not item["partial"][0]:
                break
        return item["text"]
    
        

class Text2TextGenerationModel:
    def __init__(self, model_called: str = "google/flan-t5-small"):
        self.model_called = model_called
        
    @property
    def tokenizer(self):
        return T5Tokenizer.from_pretrained(self.model_called)
    
    @property
    def model(self):
        return T5ForConditionalGeneration.from_pretrained(self.model_called).to(DEVICE)
    
    def generate(self, text: str):
        inputs_ids = self.tokenizer(text, return_tensors = "pt").input_ids
        outputs = self.model.generate(inputs_ids)
        return self.tokenizer.decode(outputs[0])
        

class SysthesiseSpeechModel:
    def __init__(self,
                 text_to_speech_model_called: str = "microsoft/speecht5_tts",
                 vocoder_model_called: str = "microsoft/speecht5_hifigan",
                 embddings_dataset_called: str = "Matthijs/cmu-arctic-xvectors"):
        self.tts_model_called = text_to_speech_model_called
        self.vcd_model_called = vocoder_model_called
        self.mbd_called = embddings_dataset_called
        
    @property
    def tts_processor(self):
        return SpeechT5Processor.from_pretrained(self.tts_model_called)
    
    @property
    def tts_model(self):
        return SpeechT5ForTextToSpeech.from_pretrained(self.tts_model_called).to(DEVICE)
    
    @property
    def vocoder(self):
        return SpeechT5HifiGan.from_pretrained(self.vcd_model_called).to(DEVICE)
    
    @property
    def speaker_embeddings(self):
        embddings_dataset = load_dataset(self.mbd_called, split="validation")
        speaker_mbd = torch.tensor(embddings_dataset[7306]["xvector"]).unsqueeze(0)
        return speaker_mbd
    
    def systhesise(self, text: str):
        inputs = self.tts_processor(text=text, return_tensors = "pt")
        speech = self.tts_model.generate_speech(
            inputs["input_ids"].to(DEVICE),
            self.speaker_embeddings.to(DEVICE),
            vocoder = self.vocoder
        )
        return speech.cpu()
    
    
    
    
class TextToImageModel:
    def __init__(self, model_called: str = "black-forest-labs/FLUX.1-dev"):
        self.model_called = model_called
        
    
    @property
    def pipe(self):
        p =  FluxPipeline.from_pretrained(self.model_called, torch_dtype=torch.bfloat16)
        if DEVICE == "cpu":
            p.enable_model_cpu_offload()
        return p
    
    
    def image(self, 
              prompt: str,
              height: int = 1024,
              width: int = 1024,
              guidance_scale: float = 3.5,
              num_inference_steps: int = 50,
              max_sequence_length: int = 512,
              generator=torch.Generator("cpu").manual_seed(0)):
        img = self.pipe(
            prompt,
            height = height,
            width = width,
            guidance_scale = guidance_scale,
            num_inference_steps = num_inference_steps,
            max_sequence_length = max_sequence_length,
            generator = generator
        ).images[0]
        img.save(f'flux-dev.png')
       

