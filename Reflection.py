"""This is the Evaluator-Optimizer pattern, where an AI agent evaluates its own output and improves it through feedback loops.

🔁 How it works:
An initial LLM generates a response or solution.

A second LLM (or the same one with a different prompt) evaluates that output against specific goals or quality criteria.

Based on the critique, the LLM refines the output.

This process repeats until the response meets the requirements or is good enough.

📌 Use Cases:
Code generation and debugging: Write code → run it → fix errors based on feedback.

Writing and editing: Draft a text → reflect on clarity/tone → revise.

Complex problem-solving: Create a plan → evaluate feasibility → improve it.

Information retrieval: Search for facts → check for completeness → refine the answer.

In short, it's a self-correcting loop that helps the model produce higher-quality results through iterative refinement."""

import os
from google import genai
import time
import asyncio
import enum
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class EvaluationStatus(enum.Enum):
    PASS = "pass"
    FAIL = "fail"
    
class EvaluationSchema(BaseModel):
    status: EvaluationStatus
    feedback: str
    response: str

def generate_poem(topic: str, feedback: str = "") -> str:
    prompt = f"Write a poem about '{topic}'."
    if feedback:
        prompt += f" Consider the following feedback: {feedback}"
    
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
    )
    
    return response.text.strip()

def evaluate(poem: str) -> EvaluationSchema:
    print("\n--- Evaluating Poem ---")
    prompt_critique = f"""Critique the following poem. Does it rhyme well? Is it exactly four lines? 
    Is it creative? Respond with PASS or FAIL and provide feedback.
    
    Poem:
    {poem}
    """
    response_critique = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt_critique,
        config={
            'response_mime_type': 'application/json',
            'response_schema': EvaluationSchema,
        },
    )
    critique = response_critique.parsed
    print(f"Evaluation Status: {critique.status}")
    print(f"Evaluation Feedback: {critique.feedback}")
    return critique

max_iterations = 3
current_iteration = 0
topic = "a man love a girl"

current_poem = generate_poem(topic)
while current_iteration <= max_iterations:
    print(f"\n--- Iteration {current_iteration + 1} ---")
    print(f"Current Poem:\n{current_poem}")
    
    evaluation = evaluate(current_poem)
    
    if evaluation.status == EvaluationStatus.PASS:
        print("Poem passed evaluation!")
        break
    
    print("Poem failed evaluation. Refining...")
    current_poem = generate_poem(topic, feedback=evaluation.feedback)
    
    if current_iteration == max_iterations:
        print("Maximum iterations reached. Final poem:")
        print(current_poem)