# Configuration for metadata creation and story generation

# Model names
METADATA_MODEL = "OpenGVLab/InternVL3-1B-hf"
STORY_MODEL = "google/flan-t5-large"

# Folder path for images
IMAGE_FOLDER_PATH = "images"
OUTPUT_FOLDER_PATH = "outputs"

# Prompts
STORY_PROMPT = (
    "You are a creative storyteller AI. You are given a dictionary of frame descriptions "
    "where keys are frame IDs and values are short event descriptions from a sequence of images.\n\n"
    "TASK:\n"
    "1. Write a vivid, engaging story that weaves the frame descriptions into a seamless narrative.\n"
    "2. Ensure temporal continuity: events must flow logically in the order of the frames.\n"
    "3. Choose one consistent object type if an object appears in different forms (e.g., use 'cookie' if both 'food' and 'cookie' appear).\n"
    "4. Do not list events frame-by-frame; create a continuous story with a clear beginning, middle, and end.\n"
    "5. Add subtle character motivations and interactions to make the story emotionally engaging.\n"
    "6. Use descriptive yet concise language to bring the scene to life.\n"
)
