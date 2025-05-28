import os
import json 
from google import genai
import enum
import asyncio
import time
from typing import Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()


client= genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class Task(BaseModel):
    task_id: str
    description: str
    assign_to: str = Field(description="Which worker type should handle this? E.g., Researcher, Writer, Coder")

class Plan(BaseModel):
    goal: str
    tasks: list[Task]
     
user_goal = "Write a short blog post about the benefits of AI agents."
prompt_planner = f"""
Create a step-by-step plan to achieve the following goal. 
Assign each step to a hypothetical worker type (Researcher, Writer).
 
Goal: {user_goal}
"""
response_planner = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=prompt_planner,
    config={
        'response_mime_type': 'application/json',
        'response_schema': Plan,
    }
)

plan = response_planner.parsed


async def execute_researcher_task(task: Task) -> Dict[str, Any]:
    """Execute a research task using the AI model."""
    prompt = f"""
    You are a professional researcher. Complete the following research task:
    
    Task: {task.description}
    
    Provide thorough, factual information with proper citations where relevant.
    """
    
    print(f"Executing research task: {task.task_id}")
    response = await client.aio.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )
    return {
        "task_id": task.task_id,
        "worker": "Researcher",
        "result": response.text.strip()
    }

async def execute_writer_task(task: Task) -> Dict[str, Any]:
    """Execute a writing task using the AI model."""
    prompt = f"""
    You are a professional writer. Complete the following writing task:
    
    Task: {task.description}
    
    Write in a clear, engaging style appropriate for a blog post.
    """
    
    print(f"Executing writing task: {task.task_id}")
    response = await client.aio.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )
    return {
        "task_id": task.task_id,
        "worker": "Writer",
        "result": response.text.strip()
    }

async def execute_coder_task(task: Task) -> Dict[str, Any]:
    """Execute a coding task using the AI model."""
    prompt = f"""
    You are a professional programmer. Complete the following coding task:
    
    Task: {task.description}
    
    Provide clean, well-commented code with explanations.
    """
    
    print(f"Executing coding task: {task.task_id}")
    response = await client.aio.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )
    return {
        "task_id": task.task_id,
        "worker": "Coder",
        "result": response.text.strip()
    }

worker_map = {
    "Researcher": execute_researcher_task,
    "Writer": execute_writer_task,
    "Coder": execute_coder_task
}

async def execute_task(task: Task) -> Dict[str, Any]:
    """Route a task to the appropriate worker function."""
    worker_function = worker_map.get(task.assign_to)
    if not worker_function:
        raise ValueError(f"No worker function defined for worker type: {task.assign_to}")
    return await worker_function(task)

async def execute_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Execute all tasks in the plan in parallel and return results."""
    start_time = time.time()
    
    tasks = []
    for task_data in plan.tasks:
        task_data = task_data.dict()
        task = Task(**task_data)
        tasks.append(execute_task(task))
        
    results = await asyncio.gather(*tasks)
    
    elapsed_time = time.time() - start_time
    print(f"All tasks completed in {elapsed_time:.2f} seconds")
    
    return {
        "goal": plan.goal,
        "results": results
    }

async def combine_results(execution_results: Dict[str, Any]) -> str:
    """Combine the results of all tasks into a final output."""
    organized_results = {}
    for result in execution_results.get("results", []):
        organized_results[result["task_id"]] = result["result"]
    result_texts = []
    for task_data in plan.tasks:
        task_id = task_data.task_id
        if task_id in organized_results:
            result_texts.append(organized_results[task_id])
    
    all_results = "\n\n".join(result_texts)

    prompt = f"""
    You are a professional blog writer. Based on the following research and writing results, create a cohesive blog post.
    {all_results}
    """
    
    response = await client.aio.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )
    
    return response.text.strip()

async def main():
    
    execution_results = await execute_plan(plan)
    print(f"Raw execution results: {json.dumps(execution_results, indent=2)}")
    
    final_output = await combine_results(execution_results)
    print("\n=== FINAL BLOG POST ===\n")
    print(final_output)

# Run the async workflow
if __name__ == "__main__":
    asyncio.run(main())

