"""
Test OpenRouter API for image generation
"""
import asyncio
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()

async def test_openrouter():
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Multi-Tool-Assistant",
        "X-Title": "Multi-Tool Assistant"
    }
    
    payload = {
        "model": "google/gemini-2.5-flash-image",
        "messages": [
            {
                "role": "user",
                "content": "Generate an image of a cat in space"
            }
        ]
    }
    
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    print(f"Testing OpenRouter API...")
    print(f"API Key: {api_key[:20]}...")
    print(f"Model: google/gemini-2.5-flash-image")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            status = response.status
            text = await response.text()
            
            print(f"\nStatus: {status}")
            print(f"Response: {text[:500]}")
            
            if status == 200:
                import json
                result = json.loads(text)
                print("\nSuccess!")
                content = result['choices'][0]['message']['content']
                print(f"Content: {content}")
                print(f"\nFull response: {json.dumps(result, indent=2)}")
            else:
                print(f"\nError: {status}")
                print(f"Details: {text}")

if __name__ == "__main__":
    asyncio.run(test_openrouter())
