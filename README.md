# AI Chatbot Evaluation

A DeepEval example that sends a prompt to the RagForge chat API and evaluates the response with Gemini using task-completion and answer-relevancy metrics.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key
- Network access to the RagForge API

## Setup

1. Create and activate a virtual environment.
2. Install the required packages:

   ```sh
   pip install deepeval requests python-dotenv langchain-google-genai
   ```

3. Create a `.env` file in this directory and set your Gemini API key:

   ```dotenv
   GOOGLE_API_KEY=your_google_api_key
   ```

## Run

From this directory, run:

```sh
python test_ragforge_agent.py
```

The script requests a chatbot response from RagForge, then evaluates it with `TaskCompletionMetric` and `AnswerRelevancyMetric`.

## Repository contents

- `test_ragforge_agent.py` — RagForge request and DeepEval evaluation example.
- `gemini_deepeval.py` — Gemini-backed DeepEval model implementation.

Local credentials, DeepEval data, virtual environments, and Python cache files are excluded by `.gitignore`.
