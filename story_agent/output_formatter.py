from .story_generator import load_story_model, generate_story, generate_title
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Builds the final output format for the story generation task.
def build_output_format(metadata_list, model_name, story_prompt):
    try:
        generator = load_story_model(model_name)
        metadata_dict = {item['frame_id']: item['scene_description'] for item in metadata_list}
        story = generate_story(generator, metadata_dict, story_prompt)
        title = generate_title(generator, story)
        characters = sorted({obj for item in metadata_list for obj in item['objects_detected']})
        events = [{"frame": item['frame_id'], "event": item['scene_description']} for item in metadata_list]
        output = {
            "title": title,
            "characters": characters,
            "summary": story,
            "events": events
        }
        return output
    except Exception as e:
        logger.error(f"Failed to build output format: {str(e)}")
        raise RuntimeError(f"Output formatting failed: {str(e)}")