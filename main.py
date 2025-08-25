import json
import os
from metadata_creation.model_loader import load_model
from metadata_creation.metadata_processor import process_images_in_folder
from story_agent.output_formatter import build_output_format
from config import METADATA_MODEL, STORY_MODEL, IMAGE_FOLDER_PATH, OUTPUT_FOLDER_PATH, STORY_PROMPT

if __name__ == "__main__":
    # Ensure output folder exists
    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

    # Step 1: Metadata creation
    processor, model, device, dtype = load_model(METADATA_MODEL)
    metadata_list = process_images_in_folder(IMAGE_FOLDER_PATH, processor, model, device, dtype)
    
    # Step 2: Story generation
    output_story = build_output_format(metadata_list, STORY_MODEL, STORY_PROMPT)
    
    # Save output to JSON file
    output_file = os.path.join(OUTPUT_FOLDER_PATH, "story_output.json")
    with open(output_file, "w") as f:
        json.dump(output_story, f, indent=2)
    print(f"Output saved to {output_file}")