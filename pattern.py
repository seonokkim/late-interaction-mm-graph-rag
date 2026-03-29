import asyncio
import os
from datetime import datetime
from pathlib import Path

import aiohttp
import hashlib
import json
from typing import List, Dict
from prompt import GRAPH_PATTERN_PROMPT
from util.request import text_request
from paths import mmqa_file
import math
import networkx as nx

API_URL = os.getenv("PATTERN_API_URL", "")
API_URL_base = os.getenv("PATTERN_API_URL_BASE", "")
CONCURRENCY = int(os.getenv("PATTERN_CONCURRENCY", "16"))
_BASE_DIR = Path(__file__).resolve().parent
_RUN_ID = os.getenv("MMGRAPHRAG_RUN_ID", datetime.now().strftime("%Y%m%d"))
JSON_FILE_PATH = os.getenv(
    "PATTERN_JSON_FILE_PATH",
    mmqa_file("MMQA_dev.jsonl"),
)
CACHE_DIR = os.getenv(
    "PATTERN_CACHE_DIR",
    str(_BASE_DIR / "result" / _RUN_ID / "phase2_pattern_cache"),
)
MAX_SAMPLES = int(os.getenv("PATTERN_MAX_SAMPLES", "0"))
DRY_RUN = os.getenv("PATTERN_DRY_RUN", "0") == "1"

async def load_json_data() -> List[Dict]:
    if "WebQA" in JSON_FILE_PATH:
        with open(JSON_FILE_PATH, "r", encoding='UTF-8') as file:
            return json.load(file)
    with open(JSON_FILE_PATH, "r", encoding='UTF-8') as file:
        return [json.loads(line) for line in file]


def hash_prompt(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


def validate_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json.load(file)
        return True
    except json.JSONDecodeError as e:
        return False
    except Exception as e:
        return False
async def make_request(session: aiohttp.ClientSession, prompt: str, data: Dict, api_base: str):
    if "WebQA" in JSON_FILE_PATH:
        cache_file = f"{CACHE_DIR}/{data['Guid']}.json"
    else:
        cache_file = f"{CACHE_DIR}/{data['qid']}.json"

    if os.path.exists(cache_file):
        if validate_json_file(cache_file):
            return
        else:
            os.remove(cache_file)

    use_dry_run = DRY_RUN or not api_base
    if use_dry_run:
        result = {
            "response": '["ENTITY","ATTRIBUTE"]##ENTITY <|> related_to <|> ATTRIBUTE<|COMPLETE|>',
            "question": data,
            "dry_run": True,
            "created_at": datetime.now().isoformat(),
        }
        out_path = f"{CACHE_DIR}/{data['Guid']}.json" if "WebQA" in JSON_FILE_PATH else f"{CACHE_DIR}/{data['qid']}.json"
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        return

    content = text_request(prompt, api_base, temperature=0.0)
    result = {"response": content, "question": data}
    if "WebQA" in JSON_FILE_PATH:
        with open(f"{CACHE_DIR}/{data['Guid']}.json", "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
    else:
        with open(f"{CACHE_DIR}/{data['qid']}.json", "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)


async def process_batch(session: aiohttp.ClientSession, template: str, json_data: List[Dict], start_index: int):
    tasks = []
    for i in range(start_index, min(start_index + CONCURRENCY, len(json_data))):
        if "WebQA" in JSON_FILE_PATH:
            prompt = template.replace("{question}", json_data[i]['Q'])
        else:
            prompt = template.replace("{question}", json_data[i]['question'])
        if i % 3 == 0:
            tasks.append(make_request(session, prompt, json_data[i], API_URL))
        else:
            tasks.append(make_request(session, prompt, json_data[i], API_URL_base))
    await asyncio.gather(*tasks)


async def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    template = GRAPH_PATTERN_PROMPT
    json_data = await load_json_data()
    if MAX_SAMPLES > 0:
        json_data = json_data[:MAX_SAMPLES]
    total_batches = math.ceil(len(json_data) / CONCURRENCY)
    if "WebQA" in JSON_FILE_PATH:
        json_data = list(json_data.values())
        val_data = []
        for data in json_data:
            if data['split'] == 'val':
                val_data.append(data)

        async with aiohttp.ClientSession() as session:
            for i in range(0, len(val_data), CONCURRENCY):
                await process_batch(session, template, val_data, i)
                current_batch = i // CONCURRENCY + 1
                progress_percentage = (current_batch / total_batches) * 100
    else:
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(json_data), CONCURRENCY):
                await process_batch(session, template, json_data, i)
                current_batch = i // CONCURRENCY + 1
                progress_percentage = (current_batch / total_batches) * 100
if __name__ == "__main__":
    asyncio.run(main())