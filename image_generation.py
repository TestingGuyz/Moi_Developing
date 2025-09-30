"""
Image Generation Module for Moi AI Assistant
Supports multiple image generation backends
"""

import os
import base64
import requests
import io
from PIL import Image
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageGenerator:
    """
    Image generation using multiple backends:
    1. Together.ai API (Stable Diffusion)
    2. Pollinations.ai (Free alternative)
    3. Hugging Face Inference API
    """
    
    def __init__(self):
        self.together_api_key = os.getenv('TOGETHER_API_KEY')
        self.hf_api_key = os.getenv('HUGGINGFACE_TOKEN')
        self.available_backends = self._check_available_backends()
        
        logger.info(f"Image generator initialized. Available backends: {self.available_backends}")
    
    def _check_available_backends(self) -> list:
        """Check which image generation backends are available"""
        backends = ['pollinations']  # Always available (free)
        
        if self.together_api_key:
            backends.append('together')
        
        if self.hf_api_key:
            backends.append('huggingface')
        
        return backends
    
    def generate_image(self, prompt: str, backend: str = 'auto', size: str = '512x512') -> Optional[bytes]:
        """
        Generate image from text prompt
        
        Args:
            prompt: Text description of the image
            backend: 'auto', 'together', 'huggingface', 'pollinations'
            size: Image size (e.g., '512x512', '1024x1024')
        
        Returns:
            Image bytes or None if failed
        """
        
        if backend == 'auto':
            # Try backends in order of preference
            for backend_name in self.available_backends:
                try:
                    return self._generate_with_backend(prompt, backend_name, size)
                except Exception as e:
                    logger.warning(f"Backend {backend_name} failed: {e}")
                    continue
            
            logger.error("All backends failed")
            return None
        else:
            return self._generate_with_backend(prompt, backend, size)
    
    def _generate_with_backend(self, prompt: str, backend: str, size: str) -> Optional[bytes]:
        """Generate image with specific backend"""
        
        if backend == 'together':
            return self._generate_together(prompt, size)
        elif backend == 'huggingface':
            return self._generate_huggingface(prompt, size)
        elif backend == 'pollinations':
            return self._generate_pollinations(prompt, size)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _generate_together(self, prompt: str, size: str) -> Optional[bytes]:
        """Generate image using Together.ai API"""
        try:
            if not self.together_api_key:
                raise ValueError("Together.ai API key not found")
            
            logger.info(f"Generating image with Together.ai: {prompt[:50]}...")
            
            # Parse size
            width, height = map(int, size.split('x'))
            
            url = "https://api.together.xyz/v1/images/generations"
            headers = {
                "Authorization": f"Bearer {self.together_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "stabilityai/stable-diffusion-xl-base-1.0",
                "prompt": prompt,
                "width": width,
                "height": height,
                "steps": 30,
                "n": 1
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            # Get image URL or base64
            if 'data' in result and len(result['data']) > 0:
                image_data = result['data'][0]
                
                if 'url' in image_data:
                    # Download image from URL
                    img_response = requests.get(image_data['url'], timeout=30)
                    img_response.raise_for_status()
                    return img_response.content
                elif 'b64_json' in image_data:
                    # Decode base64
                    return base64.b64decode(image_data['b64_json'])
            
            raise ValueError("No image data in response")
            
        except Exception as e:
            logger.error(f"Together.ai error: {e}")
            raise
    
    def _generate_huggingface(self, prompt: str, size: str) -> Optional[bytes]:
        """Generate image using Hugging Face Inference API"""
        try:
            if not self.hf_api_key:
                raise ValueError("Hugging Face API key not found")
            
            logger.info(f"Generating image with Hugging Face: {prompt[:50]}...")
            
            API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
            headers = {"Authorization": f"Bearer {self.hf_api_key}"}
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "num_inference_steps": 50,
                    "guidance_scale": 7.5
                }
            }
            
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            return response.content
            
        except Exception as e:
            logger.error(f"Hugging Face error: {e}")
            raise
    
    def _generate_pollinations(self, prompt: str, size: str) -> Optional[bytes]:
        """Generate image using Pollinations.ai (free service)"""
        try:
            logger.info(f"Generating image with Pollinations.ai: {prompt[:50]}...")
            
            # Parse size
            width, height = map(int, size.split('x'))
            
            # Pollinations.ai API
            # URL format: https://image.pollinations.ai/prompt/{prompt}?width={width}&height={height}
            
            # Encode prompt for URL
            import urllib.parse
            encoded_prompt = urllib.parse.quote(prompt)
            
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width={width}&height={height}&nologo=true"
            
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            return response.content
            
        except Exception as e:
            logger.error(f"Pollinations.ai error: {e}")
            raise
    
    def save_image(self, image_bytes: bytes, output_path: str):
        """Save image bytes to file"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.save(output_path)
            logger.info(f"Image saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False
    
    def image_to_base64(self, image_bytes: bytes) -> str:
        """Convert image bytes to base64 string"""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def generate_and_encode(self, prompt: str, backend: str = 'auto', size: str = '512x512') -> Optional[str]:
        """Generate image and return as base64 string"""
        try:
            image_bytes = self.generate_image(prompt, backend, size)
            if image_bytes:
                return self.image_to_base64(image_bytes)
            return None
        except Exception as e:
            logger.error(f"Error generating and encoding image: {e}")
            return None


# Test function
if __name__ == "__main__":
    generator = ImageGenerator()
    
    print(f"Available backends: {generator.available_backends}")
    
    # Test generation
    prompt = "A beautiful orange tree with ripe oranges"
    print(f"\nGenerating image: {prompt}")
    
    image_bytes = generator.generate_image(prompt, backend='pollinations', size='512x512')
    
    if image_bytes:
        generator.save_image(image_bytes, "test_output.png")
        print("✅ Image generated and saved as test_output.png")
    else:
        print("❌ Image generation failed")