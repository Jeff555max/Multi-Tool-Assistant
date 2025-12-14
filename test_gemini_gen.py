import asyncio
import aiohttp

async def test():
    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Multi-Tool-Assistant",
        "X-Title": "Multi-Tool Assistant"
    }
    
    payload = {
        "model": "google/gemini-2.5-flash-image",
        "messages": [
            {
                "role": "user",
                "content": "Generate an image of a beautiful ocean sunset"
            }
        ]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            print(f"Status: {response.status}")
            text = await response.text()
            print(f"Response: {text}")

asyncio.run(test())
