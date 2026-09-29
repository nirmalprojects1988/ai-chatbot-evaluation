# AI Chatbot Evaluation

A DeepEval example that sends a prompt to the chat API and evaluates the response with Gemini using task-completion and answer-relevancy metrics.

## Requirements

- Python 3.10 or newer
- A Google Gemini API key
- Network access to the API

## Setup

1. Create and activate a virtual environment.
2. Install the required packages:

   ```sh
   pip install deepeval requests python-dotenv langchain-google-genai
   ```

3. Create a `.env` file in this directory and set your Gemini API key:

   ```dotenv
   GOOGLE_API_KEY=your_google_api_key
   CHATBOT_API_URL=https://your-chatbot-api.example.com/api/chat
   CHATBOT_ORIGIN=https://your-chatbot.example.com
   BUSINESS_ID=your_business_id
   SESSION_ID=your_session_id
   VISITOR_ID=your_visitor_id
   WIDGET_ID=your_widget_id
   ```

   Replace the example values with the credentials and endpoint for your chatbot.

## Run

From this directory, run:

```sh
python test_chatbot_evaluation.py
```

The script requests a chatbot response using the configured API endpoint, then evaluates it with `TaskCompletionMetric` and `AnswerRelevancyMetric`.

## Repository contents

- `test_chatbot_evaluation.py` — chatbot API request and DeepEval evaluation example.
- `gemini_deepeval.py` — Gemini-backed DeepEval model implementation.

Local credentials, DeepEval data, virtual environments, and Python cache files are excluded by `.gitignore`.
