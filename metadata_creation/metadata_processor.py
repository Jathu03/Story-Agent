import os
import json
from PIL import Image
from datetime import datetime
from .preprocessor import prepare_inputs
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Decodes model-generated output into a human-readable string.
def output_decoder(model, processor, inputs):
    try:
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=200)

        decoded = processor.decode(
            generated[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return decoded
    except Exception as e:
        logger.error(f"Failed to decode model output: {str(e)}")
        raise RuntimeError(f"Output decoding failed: {str(e)}")

# Extracts metadata from a single image using a vision-language model.
def get_frame_metadata(image_path, processor, model, device, dtype):
    try:
        # Initialize frame metadata
        frame_id = os.path.basename(image_path)
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        logger.info(f"Processing image: {image_path}")

        # Load and preprocess image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {str(e)}")
            return {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "objects_detected": [],
                "scene_description": f"Error: Failed to load image - {str(e)}"
            }

        # Generate and decode model output
        inputs = prepare_inputs(processor, image, frame_id, timestamp, device, dtype)
        raw_output = output_decoder(model, processor, inputs)

        # Try parsing model output into JSON
        try:
            cleaned = raw_output.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON output for {image_path}: {str(e)}")
            parsed = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "objects_detected": [],
                "scene_description": raw_output.strip() or f"Error: Invalid model output - {str(e)}"
            }

        return parsed
    except Exception as e:
        logger.error(f"Error processing frame {image_path}: {str(e)}")
        return {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "objects_detected": [],
            "scene_description": f"Error: {str(e)}"
        }
# Processes all images in a folder and collects their metadata.
def process_images_in_folder(folder_path, processor, model, device, dtype):
    try:
        if not os.path.isdir(folder_path):
            logger.error(f"Invalid folder path: {folder_path}")
            raise ValueError(f"Folder {folder_path} does not exist or is not a directory")

        metadata_list = []
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(folder_path, fname)
                try:
                    meta = get_frame_metadata(image_path, processor, model, device, dtype)
                    metadata_list.append(meta)
                except Exception as e:
                    logger.warning(f"Skipping image {image_path} due to error: {str(e)}")
                    continue

        if not metadata_list:
            logger.warning(f"No valid images found in {folder_path}")
        return metadata_list
    except Exception as e:
        logger.error(f"Failed to process folder {folder_path}: {str(e)}")
        raise RuntimeError(f"Folder processing failed: {str(e)}")