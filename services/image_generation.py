"""
Image Generation Service.
Uses OpenAI DALL-E API to generate images from text prompts.
"""

import aiohttp
import aiofiles
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, IMAGE_GENERATION_MODEL, DATA_DIR
from utils.logging import logger


# Create temp directory for generated images
GENERATED_IMAGES_DIR = DATA_DIR / "generated_images"
GENERATED_IMAGES_DIR.mkdir(exist_ok=True)


async def detect_image_generation_intent(text: str, conversation_history: list = None) -> Dict[str, Any]:
    """
    Detect if user wants to generate an image using GPT.
    
    Args:
        text: User's message text
        conversation_history: Recent conversation history for context
    
    Returns:
        Dictionary with 'needs_generation' (bool) and 'prompt' (str) if detected
    """
    from services.openai_client import openai_client
    
    # Keywords that strongly suggest image generation
    strong_keywords = [
        'нарисуй', 'сгенерируй изображение', 'создай картинку', 'сделай изображение',
        'покажи как выглядит', 'визуализируй', 'нарисовать', 'создать изображение',
        'generate image', 'draw', 'create picture', 'make image', 'show me what',
        'сгенерируй картинку', 'создай изображение', 'сделай картинку'
    ]
    
    # Quick check for strong keywords
    text_lower = text.lower()
    has_strong_keyword = any(keyword in text_lower for keyword in strong_keywords)
    
    # Build detection prompt
    detection_prompt = f"""Определи, хочет ли пользователь сгенерировать изображение.

Пользовательский запрос: "{text}"

Если пользователь просит:
- нарисовать что-то
- создать/сгенерировать изображение или картинку
- визуализировать что-то
- показать как что-то выглядит

То это запрос на генерацию изображения.

Ответь СТРОГО в формате JSON:
{{
    "needs_generation": true/false,
    "prompt": "улучшенный промпт для DALL-E на английском языке (если needs_generation=true)",
    "confidence": 0.0-1.0
}}

Если это НЕ запрос на генерацию изображения (например, просто вопрос или обычный диалог), верни needs_generation: false.
"""

    try:
        # Get AI decision
        messages = [{"role": "user", "content": detection_prompt}]
        response = await openai_client.generate_text_response(messages, temperature=0.3)
        
        # Parse JSON response
        # Remove markdown code blocks if present
        response_clean = response.strip()
        if response_clean.startswith('```'):
            # Remove ```json or ``` at start and ``` at end
            lines = response_clean.split('\n')
            response_clean = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_clean
        
        result = json.loads(response_clean)
        
        # If strong keyword found but AI said no, trust the keyword
        if has_strong_keyword and not result.get('needs_generation'):
            result['needs_generation'] = True
            # If no prompt provided, use original text
            if not result.get('prompt'):
                result['prompt'] = text
        
        logger.info(f"Image generation detection: {result.get('needs_generation')} (confidence: {result.get('confidence', 0)})")
        
        return result
        
    except Exception as e:
        logger.error(f"Error detecting image generation intent: {e}")
        
        # Fallback: use keyword detection
        if has_strong_keyword:
            return {
                "needs_generation": True,
                "prompt": text,
                "confidence": 0.8
            }
        
        return {
            "needs_generation": False,
            "confidence": 0.0
        }


async def generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid"
) -> Dict[str, Any]:
    """
    Generate an image using Gemini 2.5 Flash via OpenRouter.
    
    Args:
        prompt: Text description of the image to generate
        size: Image size (ignored for Gemini)
        quality: Image quality (ignored for Gemini)
        style: Image style (ignored for Gemini)
    
    Returns:
        Dictionary with 'image_path', 'revised_prompt', and 'url'
    """
    try:
        logger.info(f"Generating image with Gemini 2.5 Flash: {prompt[:100]}...")
        
        # Prepare API request for OpenRouter
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": "Multi-Tool Assistant"
        }
        
        payload = {
            "model": IMAGE_GENERATION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": f"Generate an image: {prompt}"
                }
            ]
        }
        
        api_url = f"{OPENROUTER_BASE_URL}/chat/completions"
        
        # Make API request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                api_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenRouter API error: {error_text}")
                    raise Exception(f"OpenRouter API error: {response.status}")
                
                result = await response.json()
        
        # Extract image URL from response
        content = result['choices'][0]['message']['content']
        
        # Gemini returns markdown with image URL
        import re
        image_url_match = re.search(r'!\[.*?\]\((https://.*?)\)', content)
        if not image_url_match:
            raise Exception("No image URL found in response")
        
        image_url = image_url_match.group(1)
        
        logger.info(f"Image generated successfully with Gemini")
        
        # Download image
        image_path = await download_image(image_url)
        
        return {
            "image_path": image_path,
            "revised_prompt": prompt,
            "url": image_url,
            "original_prompt": prompt
        }
        
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise


async def download_image(url: str) -> Path:
    """
    Download image from URL and save to local file.
    
    Args:
        url: Image URL
    
    Returns:
        Path to downloaded image
    """
    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.png"
        filepath = GENERATED_IMAGES_DIR / filename
        
        # Download image
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download image: {response.status}")
                
                # Save to file
                async with aiofiles.open(filepath, 'wb') as f:
                    await f.write(await response.read())
        
        logger.info(f"Image downloaded to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        raise


async def generate_image_variations(
    image_path: Path,
    n: int = 1,
    size: str = "1024x1024"
) -> list:
    """
    Generate variations of an existing image.
    Note: This feature is not supported by Gemini model.
    
    Args:
        image_path: Path to source image
        n: Number of variations to generate (1-10)
        size: Image size
    
    Returns:
        List of paths to generated variations
    """
    logger.warning("Image variations not supported by Gemini model")
    raise NotImplementedError("Image variations not supported by Gemini 2.5 Flash model")

