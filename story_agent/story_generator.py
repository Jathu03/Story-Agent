# agent_tools.py
from langchain.tools import tool
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from agent_tools import load_story_model, generate_story_tool, generate_title_tool
from transformers import pipeline

# Global variable for model (to avoid reloading in every call)
generator = None

@tool("load_story_model", return_direct=True)
def load_story_model(model_name: str) -> str:
    """
    Loads the HuggingFace text2text generation model as a pipeline.
    Args:
        model_name (str): The HuggingFace model name (e.g., 'google/flan-t5-large')
    Returns:
        str: Confirmation message.
    """
    global generator
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto"
    )
    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.3,
    )
    return f"Model {model_name} loaded successfully."


@tool("generate_story", return_direct=True)
def generate_story_tool(inputs: dict) -> str:
    """
    Generates a story from frame descriptions.
    Args:
        inputs (dict): {
            'metadata_dict': { '1': 'desc1', '2': 'desc2', ... },
            'story_prompt': 'Write a story about...'
        }
    """
    global generator
    if generator is None:
        return "Error: Model not loaded. Please call load_story_model first."

    metadata_dict = inputs.get("metadata_dict", {})
    story_prompt = inputs.get("story_prompt", "Write a story using these frames.")

    prompt = story_prompt + "\nFrame descriptions:\n"
    for frame_id, description in sorted(metadata_dict.items()):
        prompt += f"{frame_id}: {description}\n"

    prompt += "\nNow write a compelling story that starts with the first frame, builds through the sequence, and concludes naturally, ensuring all events are connected and objects are consistent:"
    response = generator(prompt)[0]["generated_text"]
    return response.strip()


@tool("generate_title", return_direct=True)
def generate_title_tool(story_text: str) -> str:
    """
    Generates a creative title for the story.
    Args:
        story_text (str): The story text.
    """
    global generator
    if generator is None:
        return "Error: Model not loaded. Please call load_story_model first."

    prompt = (
        "Generate a concise, creative title (3-6 words) that vividly captures the main theme, "
        "character dynamics, and emotional tone of the following story. Avoid generic words (e.g., 'adventure', 'tale'), "
        "and ensure the title reflects the specific actions and relationships in the story:\n\n"
        f"{story_text}\n\nTitle:"
    )
    response = generator(prompt)[0]["generated_text"]
    return response.strip().replace('"', '')

# Using the same HuggingFace model for the LLM in the agent
hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-large", max_new_tokens=256)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Define tools
tools = [load_story_model, generate_story_tool, generate_title_tool]

# Add memory
memory = ConversationBufferMemory(memory_key="chat_history")

# Create the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

if __name__ == "__main__":
    # Example usage
    print(agent.run("Load the story model using google/flan-t5-large"))
    print(agent.run("Generate a story using metadata_dict={'1': 'A bird flies', '2': 'Fox watches the bird'} and story_prompt='Write an engaging story'"))
    print(agent.run("Now generate a title for that story"))