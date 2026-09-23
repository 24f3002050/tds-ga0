import os
import traceback
from io import StringIO
import sys
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


app = FastAPI()


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN"),
    base_url="https://aipipe.org/openai/v1"
)


def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return its output or traceback.
    """

    old_stdout = sys.stdout
    stdout = StringIO()
    sys.stdout = stdout

    try:
        exec(code)

        return {
            "success": True,
            "output": stdout.getvalue()
        }

    except Exception:
        return {
            "success": False,
            "output": traceback.format_exc()
        }

    finally:
        sys.stdout = old_stdout


def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:
    """
    Ask the LLM to identify the source-code line(s)
    responsible for the error.
    """

    prompt = f"""
Analyze the Python code and traceback below.

Identify the exact line number(s) in the submitted Python code
where the error occurred.

CODE:
{code}

TRACEBACK:
{error_traceback}

Return only the line number(s) from the user's submitted code.
Do not return traceback line numbers.
Do not guess unrelated lines.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You identify Python error line numbers. "
                    "Return structured JSON only."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "error_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "error_lines": {
                            "type": "array",
                            "items": {
                                "type": "integer"
                            }
                        }
                    },
                    "required": ["error_lines"],
                    "additionalProperties": False
                }
            }
        }
    )

    result = ErrorAnalysis.model_validate_json(
        response.choices[0].message.content
    )

    return result.error_lines


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):

    execution = execute_python_code(request.code)

    # No AI call for successful execution
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # AI is called only when execution fails
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    return {
        "error": error_lines,
        "result": execution["output"]
    }
