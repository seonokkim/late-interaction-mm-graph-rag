import asyncio
import os
import html
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import re
import aiohttp

from util.request import text_request
from prompt import EXTRACT_RELATION_PROMPT
from paths import mmqa_file

API_URL_base = os.getenv("EXTRACTION_API_URL_BASE", "")
CONCURRENCY = int(os.getenv("EXTRACTION_CONCURRENCY", "8"))
_BASE_DIR = Path(__file__).resolve().parent
_RUN_ID = os.getenv("MMGRAPHRAG_RUN_ID", datetime.now().strftime("%Y%m%d"))
CACHE_DIR = os.getenv(
    "EXTRACTION_CACHE_DIR",
    str(_BASE_DIR / "result" / _RUN_ID / "phase3_extraction_cache"),
)
QUESTION_FILE = os.getenv(
    "EXTRACTION_QUESTION_FILE",
    mmqa_file("MMQA_dev.jsonl"),
)
TEXT_FILE = os.getenv(
    "EXTRACTION_TEXT_FILE",
    mmqa_file("MMQA_texts.jsonl"),
)
PATTERN_CACHE_DIR = os.getenv(
    "EXTRACTION_PATTERN_CACHE_DIR",
    str(_BASE_DIR / "result" / _RUN_ID / "phase2_pattern_cache"),
)
MAX_QUESTIONS = int(os.getenv("EXTRACTION_MAX_QUESTIONS", "20"))
DRY_RUN = os.getenv("EXTRACTION_DRY_RUN", "1") == "1"

async def load_question_data(path:str) -> List[Dict]:
    with open(path, "r", encoding='UTF-8') as file:
        return {json.loads(line)['qid']: json.loads(line) for line in file}
async def load_text_data(path:str) -> Dict[str, Dict]:
    with open(path, "r", encoding='UTF-8') as file:
        return {json.loads(line)['id']: json.loads(line) for line in file}


def hash_prompt(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()
def cache_exists(prompt_hash: str) -> bool:
    return os.path.exists(f"{CACHE_DIR}/{prompt_hash}.json")
def validate_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json.load(file)
        return True
    except json.JSONDecodeError as e:
        return False
    except Exception as e:
        return False
def str_list(str: str) -> List[str]:
    matches = re.search(r'\[([^\]]+)\]', str)
    if matches:
        entity_types_list = matches.group(1).split(", ")
        entity_types_list = [item.strip() for item in entity_types_list]
    else:
        entity_types_list = []
    return entity_types_list
def clean_str(input: Any) -> str:
    if not isinstance(input, str):
        return input
    result = html.unescape(input.strip())
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)
async def process_graph_pattern(graph_pattern: str):
    record_delimiter = "##"
    tuple_delimiter = "<|>"
    complete_delimiter = "<|COMPLETE|>"
    records = [r.strip() for r in graph_pattern.split(record_delimiter)]
    entity_list = str_list(records[0])
    edge_type = []
    for record in records[1:]:
        edge_type.append(record.replace(complete_delimiter, ""))
    return {"type_list":entity_list, "edge_list":edge_type}
async def make_request(session: aiohttp.ClientSession, prompt: str, api_base: str,
                       text_content: str, qid: str,graph_pattern: Dict):
    if session is None:
        print("Error: session is None")
        return
    cache_file = f"{CACHE_DIR}/{qid}.json"
    if os.path.exists(cache_file):
        return
    prompt = prompt.replace("{input_text}", text_content)
    result = {}
    if DRY_RUN or not api_base:
        # Minimal deterministic pseudo extraction for downstream graph construction.
        question_text = graph_pattern.get("question_text", "")
        q_entity = clean_str(question_text).strip("?").strip().replace('"', "")[:120]
        t_entity = clean_str(text_content.split(".")[0]).strip().replace('"', "")[:120]
        if not t_entity:
            t_entity = "CONTEXT"
        result["response"] = (
            f'"entity"<|>"{q_entity}"<|>"QUESTION"<|>"query entity"##'
            f'"entity"<|>"{t_entity}"<|>"TEXT"<|>"context entity"##'
            f'"relationship"<|>"{q_entity}"<|>"{t_entity}"<|>"related context"'
        )
    else:
        result['response'] = text_request(prompt, api_base)
    result['qid'] = qid
    result['text_content'] = text_content
    result['graph_pattern'] = graph_pattern
    with open(cache_file, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
async def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    questions = await load_question_data(QUESTION_FILE)
    text_data = await load_text_data(TEXT_FILE)
    question_items = list(questions.items())[:MAX_QUESTIONS] if MAX_QUESTIONS > 0 else list(questions.items())
    k = 1
    async with aiohttp.ClientSession() as session:
        tasks = []
        for key, question in question_items:
            qid = key
            pattern_file = os.path.join(PATTERN_CACHE_DIR, f"{qid}.json")
            graph_pattern = {"type_list": [], "edge_list": [], "question_text": question.get("question", "")}
            if os.path.exists(pattern_file):
                with open(pattern_file, "r", encoding="utf-8") as file:
                    graph_pattern_raw = json.load(file)
                graph_pattern = await process_graph_pattern(graph_pattern_raw.get("response", ""))
                graph_pattern["question_text"] = question.get("question", "")
            new_template = EXTRACT_RELATION_PROMPT.replace("{Graph_pattern}",
                                            "Entity types: [" + ",".join(graph_pattern["type_list"]) + "]\n" + "\n".join(
                                                graph_pattern["edge_list"]))
            text_doc_ids = question.get("metadata", {}).get("text_doc_ids", [])[:2]
            joined_texts = []
            for text_doc_id in text_doc_ids:
                entry = text_data.get(text_doc_id)
                if entry and entry.get("text"):
                    joined_texts.append(entry["text"][:1000])
            text_content = "\n".join(joined_texts)
            api_url = API_URL_base
            if not text_doc_ids:
                text_doc_ids = [f"{qid}_NO_TEXT"]
            for text_doc_id in text_doc_ids:
                task_qid = f"{qid}_{text_doc_id}"
                task = make_request(session, new_template, api_url, text_content, task_qid, graph_pattern)
                tasks.append(task)
                if len(tasks) >= CONCURRENCY:
                    await asyncio.gather(*tasks)
                    tasks = []

            k += 1
            print(f'{datetime.now()}:{k / len(question_items)}')
        if len(tasks) > 0:
            await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())