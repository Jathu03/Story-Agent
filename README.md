# Image Captioning and Story Generation Project

This project processes a folder of images to generate metadata describing the actions in each image and then creates a cohesive story based on the sequence of metadata.

## Project Structure

````
project_root/
├── metadata_creation/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── preprocessor.py
│   ├── metadata_processor.py
├── story_agent/
│   ├── __init__.py
│   ├── story_generator.py
│   ├── output_formatter.py
├── config.py
├── main.py
├── requirements.txt
├── README.md
├── images
    ├── story1
│   ├── story2

## Prerequisites
- Python 3.8+
- A folder named `images` containing story images

## Installation
1. Clone or download this repository.
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
````

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dependencies

- `torch`: For model operations and tensor computations
- `timm`: For vision model utilities
- `transformers`: For loading pre-trained models and processors
- `pillow`: For image processing

## Usage

1. Place your images in the `images` folder (or update `IMAGE_FOLDER_PATH` in `config.py`).
2. Configure settings in `config.py`:
   - `METADATA_MODEL`: Model for image captioning
   - `STORY_MODEL`: Model for story generation
   - `IMAGE_FOLDER_PATH`: Path to the images folder
   - `OUTPUT_FOLDER_PATH`: Path where the JSON output will be saved
   - `STORY_PROMPT`: Prompt for story generation
3. Run the main script:
   ```bash
   python main.py
   ```
4. The script will:
   - Load the metadata model (`OpenGVLab/InternVL3-1B-hf`) and process images to generate metadata.
   - Load the story model (`google/flan-t5-large`) and generate a story based on the metadata.
   - Save the result as a JSON file in the folder specified by `OUTPUT_FOLDER_PATH` in `config.py` (default: `outputs/story_output.json`).

## Notes

- Ensure the `images` folder exists and contains valid image files.
- The output folder is specified in `config.py` (`OUTPUT_FOLDER_PATH`). Ensure it is a valid path.
- The project uses GPU if available; otherwise, it falls back to CPU.
- The output JSON is saved as `story_output.json` in the specified output folder.

```

```
