import base64
import os
import re

import requests
import google.generativeai as genai
import PIL.Image
import time
import torch
import io
from dotenv import load_dotenv


load_dotenv()


def load_pretrained_model_with_fallback(model_cls, model_ref: str, **kwargs):
    """
    Prefer CUDA auto placement, then fall back to CPU.
    """
    if torch.cuda.is_available():
        try:
            return model_cls.from_pretrained(model_ref, device_map="auto", **kwargs)
        except Exception as exc:
            print(f"CUDA load failed for {model_ref}; falling back to CPU. reason={exc}")
    return model_cls.from_pretrained(model_ref, device_map={"": "cpu"}, **kwargs)


def text_request(prompt: str, api_base: str, temperature=0.0):
    # Priority: explicit arg -> generic LLM_* from .env -> legacy SILICONFLOW_* -> default.
    if api_base:
        api_url = api_base
    elif os.getenv("LLM_BASE_URL"):
        api_url = os.getenv("LLM_BASE_URL", "").rstrip("/") + "/chat/completions"
    else:
        api_url = os.getenv("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1/chat/completions")
    api_key = os.getenv("LLM_API_KEY", "") or os.getenv("SILICONFLOW_API_KEY", "")
    model_name = os.getenv("LLM_MODEL", "Qwen2.5-72B-Instruct")
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "max_tokens": 512,
        "stop": None,
        "temperature": temperature,
        "top_p": 0.1,
        "n": 1,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.request("POST", url=api_url, json=payload, headers=headers, timeout=90)
    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]
        return content
    elif response.status_code == 429:
        time.sleep(10)
        return text_request(prompt, api_base, temperature)
    else:
        raise RuntimeError(f"text_request failed: status={response.status_code}, body={response.text}")

def llava_image_request(prompt, path, api_base, temperature=0.0):
    with open(path, 'rb') as image_file:
        imagedata = image_file.read()
    encoded_image = base64.b64encode(imagedata).decode('utf-8')
    images = [encoded_image]
    payload = {
        "model": "llava:34b",  # 使用llava:34b模型
        "num_ctx": 10240,
        "temperature": temperature,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images,
            }
        ],
        "stream": False,
    }

    response = requests.post(api_base, json=payload)

    if response.status_code == 200:
        return response.json()['message']['content']
    else:
        print(f"请求失败，状态码: {response.status_code}")
        return None

def gemini_select(prompt, image_paths, api_key=""):

   try:
       api_key = api_key or os.getenv("GEMINI_API_KEY", "")
       images = [PIL.Image.open(path) for path in image_paths]

       with requests.Session() as session:
           genai.configure(
               api_key=api_key,
               transport="rest",
               client_options={
                   "api_endpoint": "https://generativelanguage.googleapis.com"
               }
           )
           model = genai.GenerativeModel(model_name="gemini-1.5-flash")
           content = [prompt] + images
           response = model.generate_content(content)
           return response.text

   except Exception:
       return None

   finally:
       pass


