import asyncio
import aiohttp

async def test_image_generation():
    api_key = "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"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Multi-Tool-Assistant",
        "X-Title": "Multi-Tool Assistant"
    }
    
    payload = {
        "model": "black-forest-labs/flux-schnell",
        "messages": [
            {
                "role": "user",
                "content": "Generate an image: beautiful ocean sunset"
            }
        ]
    }
    
    api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    print(f"Sending request to: {api_url}")
    print(f"Model: {payload['model']}")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            response_text = await response.text()
            print(f"\nStatus: {response.status}")
            print(f"Response: {response_text[:1000]}")
            
            if response.status == 200:
                result = await response.json()
                if 'choices' in result:
                    content = result['choices'][0]['message']['content']
                    print(f"\nContent: {content}")

if __name__ == "__main__":
    asyncio.run(test_image_generation())
