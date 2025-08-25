import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_checkpoint):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        logger.info(f"Loading model {model_checkpoint} on {device} with dtype {dtype}")

        processor = AutoProcessor.from_pretrained(model_checkpoint, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_checkpoint,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(device).eval()

        logger.info("Model and processor loaded successfully")
        return processor, model, device, dtype
    except Exception as e:
        logger.error(f"Failed to load model {model_checkpoint}: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")