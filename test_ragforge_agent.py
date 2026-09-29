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
# 1. RagForge API
# --------------------------------------------------

url = "https://api.ragforgeai.com/api/chat"

headers = {
    "Content-Type": "application/json",
    "Origin": "https://www.ragforgeai.com",
}

message = "What services does RagForge AI provide?"

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
# 2. Get actual response from RagForge
# --------------------------------------------------

actual_output = data["reply"]

print("\nAgent Response:")
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

metric = TaskCompletionMetric(
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
        metric,
        answer_relevancy_metric,
    ],
)
