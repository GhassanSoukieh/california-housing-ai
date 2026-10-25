import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


class ExplanationError(RuntimeError):
    """Raised when Ollama cannot produce a usable explanation."""


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

    try:
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
        explanation = response.json().get("response")
    except requests.exceptions.HTTPError as exc:
        raise ExplanationError(
            "The explanation service rejected the request. Verify that the "
            "llama3.2 model is installed in Ollama."
        ) from exc
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        raise ExplanationError(
            "The explanation service is unavailable. Make sure Ollama is running "
            "and the llama3.2 model is installed."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise ExplanationError(
            "The explanation service could not complete the request."
        ) from exc
    except (ValueError, AttributeError, TypeError) as exc:
        raise ExplanationError(
            "The explanation service returned an invalid response."
        ) from exc

    if not isinstance(explanation, str) or not explanation.strip():
        raise ExplanationError("The explanation service returned no explanation.")

    return explanation
