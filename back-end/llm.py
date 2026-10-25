import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def explain_prediction(housing_data: dict, prediction: dict):

    prompt = f"""
    You are a real estate assistant.

    Here is the housing information:
    {housing_data}

    The AI model predicted:
    ${prediction["price_usd"]:,.2f}

    Explain this predicted median housing value
    in simple English.

    Explain what the input features represent.
    Do not invent reasons why the model produced this value.
    """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"]