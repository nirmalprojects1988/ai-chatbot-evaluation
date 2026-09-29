import os
import sys
import requests

from dotenv import load_dotenv


# --------------------------------------------------
# 0. Project path setup
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

# Allow Python to find the local Gemini evaluator.
sys.path.insert(0, PROJECT_ROOT)

# Load .env from this repository's root.
ENV_PATH = os.path.join(
    PROJECT_ROOT,
    ".env"
)

load_dotenv(ENV_PATH)


# --------------------------------------------------
# DeepEval imports
# --------------------------------------------------

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    TaskCompletionMetric,
)

from gemini_deepeval import GeminiEvaluator


# --------------------------------------------------
# 1. Chatbot API
# --------------------------------------------------

url = os.getenv("CHATBOT_API_URL")
origin = os.getenv("CHATBOT_ORIGIN")

required_env_vars = {
    "CHATBOT_API_URL": url,
    "CHATBOT_ORIGIN": origin,
    "BUSINESS_ID": os.getenv("BUSINESS_ID"),
    "SESSION_ID": os.getenv("SESSION_ID"),
    "VISITOR_ID": os.getenv("VISITOR_ID"),
    "WIDGET_ID": os.getenv("WIDGET_ID"),
}

missing_vars = [
    name
    for name, value in required_env_vars.items()
    if not value
]

if missing_vars:
    raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing_vars)}"
    )


headers = {
    "Content-Type": "application/json",
    "Origin": origin,
}

message = "What services does the chatbot provide?"

payload = {
    "business_id": os.getenv("BUSINESS_ID"),
    "message": message,
    "session_id": os.getenv("SESSION_ID"),
    "visitor_id": os.getenv("VISITOR_ID"),
    "language": "en",
    "widget_id": os.getenv("WIDGET_ID"),
}

response = requests.post(
    url,
    headers=headers,
    json=payload,
    timeout=30,
)

response.raise_for_status()

data = response.json()


# --------------------------------------------------
# 2. Get actual chatbot response
# --------------------------------------------------

actual_output = data["reply"]

print("\nChatbot Response:")
print(actual_output)


# --------------------------------------------------
# 3. Create DeepEval test case
# --------------------------------------------------

test_case = LLMTestCase(
    input=message,
    actual_output=actual_output,
)


# --------------------------------------------------
# 4. Gemini evaluator
# --------------------------------------------------

gemini_evaluator = GeminiEvaluator(
    model="gemini-3-flash-preview",
)


# --------------------------------------------------
# 5. Metrics
# --------------------------------------------------

task_completion_metric = TaskCompletionMetric(
    threshold=0.4,
    model=gemini_evaluator,
)

answer_relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=gemini_evaluator,
)


# --------------------------------------------------
# 6. Evaluate
# --------------------------------------------------

evaluate(
    test_cases=[test_case],
    metrics=[
        task_completion_metric,
        answer_relevancy_metric,
    ],
)
