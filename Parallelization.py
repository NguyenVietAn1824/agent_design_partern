import os
from google import genai
import time
import asyncio

from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


topic = "a friendly robot exploring a jungle"

prompts = [
f"Write a short, adventurous story idea about {topic}.",
f"Write a short, funny story idea about {topic}.",
f"Write a short, mysterious story idea about {topic}."
]


async def generate_content_async(prompt, model='gemini-2.0-flash'):
    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config={
            'response_mime_type': 'application/json',
        }
    )
    return response.text.strip()

async def parallel_generate(prompts, model='gemini-2.0-flash'):
    start_time = time.time()
    tasks = [generate_content_async(prompt, model) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    elapsed_time = time.time() - start_time
    print(f"Parallel generation completed in {elapsed_time:.2f} seconds")
    
    stored_results = "\n".join(f"Idea {i + 1}: {result}" for i, result in enumerate(results))
    
    aggregated_prompt = f"Combime the following ideas into a single, coherent response:\n{stored_results}"
    aggregated_response = await client.aio.models.generate_content(
        model=model,
        contents=aggregated_prompt,
        config={
            'response_mime_type': 'application/json',
        }
    )
    return aggregated_response.text.strip()

async def main():
    result = await parallel_generate(prompts)
    print(f"Aggregated Response: {result}")

if __name__ == "__main__":
    asyncio.run(main())