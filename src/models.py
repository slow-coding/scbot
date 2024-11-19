from langchain_openai import ChatOpenAI


def create_model(model_name):
    return ChatOpenAI(
        temperature=0.4,
        base_url="http://localhost:1234/v1",
        model=model_name,
        api_key="None",  # Assuming an empty string for the API key if it's not meant to be used
    )
