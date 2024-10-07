
import torch

from IPython.display import Audio

def device():
    """
    Determine the device of the local computer or machince
    to apply for the model.
    """
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def translate(audio, pipe, max_new_tokens:int, generate_kwargs: dict):
    """
    Convert audio to text.
    
    Parameters:
    ----------
    audio:
    
    pipe: 
    
    max_new_tokens: int
        The max new tokens which use it to limit the length of new tokens.
        
    generate_kwargs: dict
        Example: generate_kwargs = {'task': 'translate'}
    
    Returns:
    --------
    str
    """
    outputs = pipe(audio, 
                   max_new_tokens = max_new_tokens,
                   generate_kwargs = generate_kwargs)
    return outputs["text"]

