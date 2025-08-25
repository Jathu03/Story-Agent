from PIL import Image
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_inputs(processor, image, frame_id, timestamp, device, dtype):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            "This is a frame from a continuous story. Focus strictly on: "
                            "1. All main objects/subjects present "
                            "2. Their current actions (especially if something is falling, dropping, or in motion) "
                            "3. Any immediate interactions between objects "
                            "Describe only what is actively happening in this exact moment. "
                            "If something is falling or being released, make that the focus. "
                            "Use simple present tense verbs only. "
                            "No adjectives, background details, or static descriptions. "
                            "Format exactly as: "
                            "{"
                            f'"frame_id": "{frame_id}", '
                            f'"timestamp": "{timestamp}", '
                            '"objects_detected": ["subject1", "subject2", "object1"], '
                            '"scene_description": "subject1 verb object1, subject2 verb"'
                            "}"
                            "\nExample outputs:"
                            '{"scene_description": "crow drops peanut"}'
                            '{"scene_description": "ball rolls on ground, child reaches for it"}'
                            '{"scene_description": "car turns left, pedestrian crosses road"}'
                        )
                    }
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        # Move all tensors to the right device and dtype
        for k, v in inputs.items():
            if torch.is_tensor(v):
                try:
                    if k == 'input_ids':
                        inputs[k] = v.to(device=device)
                    else:
                        inputs[k] = v.to(device=device, dtype=dtype)
                except Exception as e:
                    logger.error(f"Failed to move tensor {k} to device {device}: {str(e)}")
                    raise RuntimeError(f"Tensor conversion failed: {str(e)}")

        return inputs
    except Exception as e:
        logger.error(f"Failed to prepare inputs for frame {frame_id}: {str(e)}")
        raise RuntimeError(f"Input preparation failed: {str(e)}")