import base64
import asyncio
import hashlib
import io
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
import chromadb
import requests
import networkx as nx
import torch

from prompt import *

from util.request import (
    llava_image_request,
    text_request,
    gemini_select,
    load_pretrained_model_with_fallback,
)
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

from paths import (
    default_colembed_model_ref,
    default_log_dir,
    mmqa_embedding_dir,
    mmqa_file,
    mmqa_images_dir,
)

API_URL = os.getenv("INFERENCE_API_URL", "")
DRY_RUN = os.getenv("INFERENCE_DRY_RUN", "1") == "1"
_BASE_DIR = Path(__file__).resolve().parent
_RUN_ID = os.getenv("MMGRAPHRAG_RUN_ID", datetime.now().strftime("%Y%m%d"))
GRAPH_DIR = os.getenv(
    "INFERENCE_GRAPH_DIR",
    str(_BASE_DIR / "result" / _RUN_ID / "phase4_graphs_real"),
)
QUESTION_FILE = os.getenv(
    "INFERENCE_QUESTION_FILE",
    mmqa_file("MMQA_dev.jsonl"),
)
OUTPUT_JSON = os.getenv(
    "INFERENCE_OUTPUT_JSON",
    str(_BASE_DIR / "result" / _RUN_ID / "phase5_predictions_real.json"),
)
INFERENCE_RETRIEVAL_JSON = os.getenv("INFERENCE_RETRIEVAL_JSON", "")
MAX_QUESTIONS = int(os.getenv("INFERENCE_MAX_QUESTIONS", "20"))
LOG_ROOT = Path(os.getenv("MMGRAPHRAG_LOG_DIR", str(default_log_dir())))
_LOG_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_FILE_PATH = LOG_ROOT / f"{_LOG_TIMESTAMP}_colembed_inference.log"

model = None
processor = None
logger = logging.getLogger("late_interaction_mm_graph_rag")


def _default_retrieval_json_path(output_json_path: str) -> str:
    if output_json_path.endswith(".json"):
        return output_json_path[:-5] + "_retrieval.json"
    return output_json_path + "_retrieval.json"


def extract_ranked_source_ids_from_graph(graph: nx.Graph, top_k: int = 10):
    ranked = []
    seen = set()
    for node_name, node_data in graph.nodes(data=True):
        source_id = node_data.get("source_id")
        if not source_id:
            continue
        sid = str(source_id)
        if sid in seen:
            continue
        seen.add(sid)
        ranked.append(
            {
                "id": sid,
                "score": float(graph.degree[node_name]),
            }
        )
    ranked.sort(key=lambda x: (-x["score"], x["id"]))
    return ranked[:top_k]


def setup_logger() -> logging.Logger:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(_LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    logger.info("Logger initialized. log_file=%s", _LOG_FILE_PATH)
    return logger

def resolve_vision_model_ref() -> str:
    return default_colembed_model_ref()


def use_trust_remote_code() -> bool:
    return os.getenv("COLEMBED_TRUST_REMOTE_CODE", "1").strip().lower() not in {"0", "false", "no"}


def use_score_debug_log() -> bool:
    return os.getenv("COLEMBED_DEBUG_SCORES", "0").strip().lower() in {"1", "true", "yes"}


def ensure_colembed_retrieval_api(loaded_model) -> None:
    required = ("forward_queries", "forward_images", "get_scores")
    missing = [name for name in required if not hasattr(loaded_model, name)]
    if missing:
        raise RuntimeError(
            f"Colembed retrieval API missing required methods: {missing}. "
            "Model must support forward_queries/forward_images/get_scores."
        )


def probe_model_output_contract(model_ref: str = "", device_map: str = "auto") -> dict:
    """
    Phase-A contract probe:
    checks whether model forward outputs expose
    - text_embeds
    - image_embeds
    - logits_per_text
    """
    ref = model_ref or resolve_vision_model_ref()
    trust_remote_code = use_trust_remote_code()
    setup_logger()
    logger.info(
        "Phase A probe started. model_ref=%s device_map=%s trust_remote_code=%s",
        ref,
        device_map,
        trust_remote_code,
    )
    if device_map == "auto":
        local_model = load_pretrained_model_with_fallback(
            AutoModel, ref, trust_remote_code=trust_remote_code
        )
    else:
        local_model = AutoModel.from_pretrained(
            ref, device_map=device_map, trust_remote_code=trust_remote_code
        )
    local_processor = AutoProcessor.from_pretrained(ref, trust_remote_code=trust_remote_code)
    dummy_image = Image.new("RGB", (32, 32), color=(255, 255, 255))
    inputs = local_processor(
        text=["contract probe"],
        images=[dummy_image],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    # Keep probe tensors on same device as model to avoid cuda/cpu mismatch.
    model_device = next(local_model.parameters()).device
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model_device)
    with torch.no_grad():
        outputs = local_model(**inputs)
    result = {
        "model_ref": ref,
        "has_forward_queries": hasattr(local_model, "forward_queries"),
        "has_forward_images": hasattr(local_model, "forward_images"),
        "has_get_scores": hasattr(local_model, "get_scores"),
        "has_text_embeds": hasattr(outputs, "text_embeds"),
        "has_image_embeds": hasattr(outputs, "image_embeds"),
        "has_logits_per_text": hasattr(outputs, "logits_per_text"),
    }
    logger.info("Phase A probe result: %s", json.dumps(result, ensure_ascii=False))
    return result

def get_answer(image_title:str,  image_path: str, question: str):
    prompt = IMAGE_QA_PROMPT
    try:
        prompt = prompt.replace("{question}", question).replace("{title}", image_title)
        return llava_image_request(prompt, image_path, "")
    except Exception as e:
        print(e)
        return None

def str_to_dict_list(str_dict_list):
    try:
        # 使用 json.loads 将字符串转换为列表或字典
        result_list = json.loads(str_dict_list)
        return result_list
    except json.JSONDecodeError as e:
        return None

def graph_to_graphml_str(graph):
    with io.BytesIO() as byte_output:
        nx.write_graphml(graph, byte_output)
        byte_output.seek(0)
        graphml_str = byte_output.read().decode('utf-8')
    return graphml_str

def graph_to_str(graph):
    output = []
    text_nodes = []
    image_nodes = []
    table_nodes = []

    for node in graph.nodes(data=True):
        node_id = node[0]
        node_data = node[1]
        node_info = {
            'id': node_id,
            'name': node_data.get('entity_name', ''),
            'type': node_data.get('type', ''),
            'description': node_data.get('description', '')
        }
        if node_id.endswith('IMAGE'):
            image_nodes.append(node_info)
        elif node_id.endswith('TABLE'):
            table_nodes.append(node_info)
        else:
            text_nodes.append(node_info)
    output.append("======= BEGIN: TEXT NODES BLOCK =======")
    for node in text_nodes:
        if node['name'] and node['type']:
            output.append(f"Name: {node['name']}")
            output.append(f"Type: {node['type']}")
            output.append(f"Description: {node['description']}")
            output.append("---")
    output.append("======= END: TEXT NODES BLOCK =======")
    output.append("")

    output.append("======= BEGIN: IMAGE NODES BLOCK =======")
    for node in image_nodes:
        if node['name']:
            output.append(f"Name: {node['name']}")
            output.append(f"Type: image")
            output.append(f"Description: {node['description']}")
            output.append("---")
    output.append("======= END: IMAGE NODES BLOCK =======")
    output.append("")

    output.append("======= BEGIN: TABLE NODES BLOCK =======")
    for node in table_nodes:
        if node['name']:
            output.append(f"Name: {node['name']}")
            output.append(f"Type: table")
            output.append(f"Description: {node['description']}")
            output.append("---")
    output.append("======= END: TABLE NODES BLOCK =======")
    output.append("")

    output.append("======= BEGIN: RELATIONSHIPS BLOCK =======")
    for edge in graph.edges(data=True):
        source_node = graph.nodes[edge[0]]
        target_node = graph.nodes[edge[1]]
        edge_data = edge[2]

        if source_node.get('entity_name') and target_node.get('entity_name'):
            output.append(f"Node 1 Name: {source_node['entity_name']}")
            if source_node.get('type') and source_node.get('type') != 'unspecified':
                output.append(f"Node 1 Type: {source_node['type']}")
            output.append(f"Node 2 Name: {target_node['entity_name']}")
            if target_node.get('type') and target_node.get('type') != 'unspecified':
                output.append(f"Node 2 Type: {target_node['type']}")
            if edge_data.get('description') and edge_data.get('description') != 'unspecified':
                output.append(f"Relationship between Node 1 and Node 2: {edge_data['description']}")
            output.append("----------")
    output.append("======= END: RELATIONSHIPS BLOCK =======")

    return '\n'.join(output)

def load_image_data(text_file_path):
    url_dict = {}
    with open(text_file_path, "r", encoding='UTF-8') as file:
        for line in file:
            data = json.loads(line)
            id = data['id']
            if id not in url_dict:
                url_dict[id] = []
            url_dict[id].append(data)
    return url_dict

def cache_exists(path):
    return os.path.exists(f"{path}")

def validate_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json.load(file)
        return True
    except json.JSONDecodeError as e:
        return False
    except Exception as e:
        return False

def get_embeding(prompt, model="bge-m3"):
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    cache_file = os.path.join(mmqa_embedding_dir(), f"{prompt_hash}.json")
    if cache_exists(cache_file):
        if validate_json_file(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as file:
                    result = json.load(file)
                    return result['response']['embedding']
            except:
                pass
    data = {
        "model": model,
        "prompt": prompt
    }
    response = requests.post("", headers={"Content-Type": "application/json"},
                             data=json.dumps(data)).json()
    with open(cache_file, "w", encoding="utf-8") as file:
        file.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False, indent=2))
    return response["embedding"]

def text_to_image_feature(image_paths, texts, n=3):
    global model, processor
    if use_score_debug_log():
        setup_logger()
    if model is None or processor is None:
        model_ref = resolve_vision_model_ref()
        trust_remote_code = use_trust_remote_code()
        model = load_pretrained_model_with_fallback(
            AutoModel, model_ref, trust_remote_code=trust_remote_code
        )
        ensure_colembed_retrieval_api(model)
        processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=trust_remote_code)
    images = [Image.open(image_path).convert('RGB') for image_path in image_paths]
    if not texts:
        return []
    with torch.no_grad():
        # Phase B enforcement: both query and image are encoded by the same Colembed model.
        query_embeddings = model.forward_queries(texts, batch_size=max(1, len(texts)))
        image_embeddings = model.forward_images(images, batch_size=max(1, len(images)))
        scores = model.get_scores(query_embeddings, image_embeddings)
    # Keep legacy return contract: indices for the first text query.
    scores_1d = scores[0]
    k = min(n, int(scores_1d.shape[0]))
    if k <= 0:
        return []
    topk = torch.topk(scores_1d, k=k, dim=-1)
    indices = topk.indices.tolist()
    if use_score_debug_log():
        values = topk.values.detach().float().cpu()
        logger.info(
            "Colembed score debug | texts=%d images=%d score_shape=%s topk_indices=%s topk_min=%.6f topk_max=%.6f",
            len(texts),
            len(images),
            tuple(scores.shape),
            indices,
            float(values.min().item()),
            float(values.max().item()),
        )
    return indices

def extract_answer_list(text, answer_pattern=r'<\|Answer\|>([\s\S]*?)<\|\\Answer\|>'):
    output = text
    answers = []
    match = re.findall(answer_pattern, output)
    for item in match:
        item = item.strip()
        try:
            item_list = str_to_dict_list(item)
            if item_list is not None:
                answers.append(item_list)
        except (ValueError, SyntaxError):
            pass

    return answers[0] if answers else []

async def get_image_feature(image_feature_prompt, qwen_api, G, image_data_dict):
    image_first_list = []
    response = text_request(image_feature_prompt, qwen_api)
    image_feature_list = extract_answer_list(response, answer_pattern=r'<\|Answer\|>([\s\S]*?)<\|\\Answer\|>')
    if len(image_feature_list) == 0:
        return []

    # 收集所有图片节点信息
    images_index = []
    images = []
    for node in G.nodes(data=True):
        try:
            if 'type' not in node[1]:
                continue
            if node[1]['type'] == "IMAGE":
                images_index.append(node[0])
                try:
                    images.append(
                        os.path.join(
                            mmqa_images_dir(),
                            image_data_dict[node[1]['source_id']][0]['path'],
                        ))
                except Exception as ex:
                    print(f"Error getting image path: {ex}")
                    continue
        except Exception as e:
            print(f"Error processing node: {e}")
            continue


    if len(image_feature_list) != 0 and len(images) > 0:
        for item in image_feature_list:
            try:
                index = text_to_image_feature(images, [item], n=3)  # 使用单个文本
                if not index:
                    continue

                image_first = [images_index[i] for i in index]

                # 准备代理选择
                image_first_path_list = []
                for img_id in image_first:
                    for node in G.nodes(data=True):
                        try:
                            if 'type' not in node[1]:
                                continue
                            if node[1]['type'] == "IMAGE" and node[0] == img_id:
                                image_path = os.path.join(
                                    mmqa_images_dir(),
                                    image_data_dict[node[1]['source_id']][0]['path'],
                                )
                                image_first_path_list.append(image_path)
                                break
                        except Exception as exc:
                            print(f"Error getting proxy image path: {exc}")
                            continue

                try:
                    # 直接进行代理选择
                    if len(image_first_path_list) >= 3:
                        temp_prompt = SELECT_IMAGE_PROMPT.replace("{description}", item)
                        result = gemini_select(temp_prompt, image_first_path_list)
                        result = result.lower().strip()
                    else:
                        result = "answer:1"


                    # 使用更灵活的检查方式
                    if "answer:1" in result.replace(" ", ""):
                        selected_index = 0
                    elif "answer:2" in result.replace(" ", ""):
                        selected_index = 1
                    elif "answer:3" in result.replace(" ", ""):
                        selected_index = 2
                    else:
                        image_first_list = []
                        return image_first_list
                    final_image = image_first[selected_index]
                    for node in G.nodes(data=True):
                        try:
                            if 'type' not in node[1]:
                                continue
                            if node[1]['type'] == "IMAGE" and node[0] == final_image:
                                node[1]['description'] = f'This image matches the description in the question "{item}"'
                                break
                        except Exception as exc:
                            continue

                    image_first_list.append(final_image)

                except Exception as ex:
                    print(f"Error in proxy selection: {ex}")
                    if image_first:
                        image_first_list.append(image_first[0])

            except Exception as e:
                print(f"Error processing feature item: {e}")
                continue

    return image_first_list

async def get_Q_entity(collection, nodes):
    for name, node in nodes:
        if node['description'] == "":
            continue
        embedding = get_embeding(node['description'], model="bge-m3")
        collection.add(
            ids=[name],
            embeddings=[embedding],
            documents=[name]
        )
    query_embeddings = [get_embeding(question['question'].strip("\""), model="bge-m3")]
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=2
    )

    question_entity = list(set([item for sublist in results['documents'] for item in sublist]))
    return question_entity

async def get_head_entity(collection, nodes):
    for name, node in nodes:
        embedding = get_embeding(name)
        collection.add(
            ids=[name],
            embeddings=[embedding],
            documents=[name])
    head_node_prompt = HEAD_NODE_PROMPT.replace("{question}", question['question'])
    response = text_request(head_node_prompt, API_URL)
    start_list = extract_answer_list(response, answer_pattern=r'<\|Answer\|>([\s\S]*?)<\|\\Answer\|>')
    query_embeddings = []
    for item in start_list:
        query_embedding = get_embeding(item)
        query_embeddings.append(query_embedding)
    if len(query_embeddings) == 0:
        query_embeddings.append(get_embeding(question['question']))
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=2
    )

    first_entity = list(set([item for sublist in results['documents'] for item in sublist]))
    return first_entity

async def main(question, image_data_dict):
    number_api = {"k":0, "image_k":0, "table_k":0, "image_select":0}
    qid = question["qid"]
    client = chromadb.Client()
    graph_path = os.path.join(GRAPH_DIR, f'{question["qid"]}_graph.graphml')
    G = nx.read_graphml(graph_path)

    nodes = G.nodes(data=True)
    image_feature_prompt = IMAGE_FEATURE_PROMPT.replace("{question}", question['question'])
    try:
        client.delete_collection(name="Q_collection")
    except:
        pass
    Q_collection = client.create_collection(name="Q_collection")

    try:
        client.delete_collection(name="entity")
    except:
        pass
    entity_collection = client.create_collection(name="entity")
    image_first_list, question_entity, head_entity = await asyncio.gather(
        get_image_feature(image_feature_prompt, API_URL, G, image_data_dict),
        get_Q_entity(Q_collection, nodes),
        get_head_entity(entity_collection, nodes)
    )
    number_api['image_select'] = len(image_first_list)
    first_entity = list(set(head_entity + question_entity + image_first_list))

    imageQ_prompt = IMAGEQ_PROMPT
    table_qa_prompt = TABLE_QA_PROMPT
    determine_answer_prompt = DETERMINE_ANSWER_PROMPT
    llm_answer_prompt = LLM_ANSWER_PROMPT

    essential_graph = nx.Graph()
    irrelation_graph = []
    for entity in first_entity:
        essential_graph.add_node(entity, **nodes[entity])
    k = 0
    sourse_id = []
    for node, data in essential_graph.nodes(data=True):
        if data["type"] == "IMAGE":
            sourse_id.append(data['source_id'])
    table_count = 0
    while k < 5:
        temp = determine_answer_prompt.replace("{question}",
                                               question['question'])
        temp = temp.replace("{GraphML}", graph_to_str(essential_graph))
        result = text_request(temp, API_URL)
        try:
            answer = re.search(r"Answer:\s*(Yes|No)", result, re.IGNORECASE).group(1).capitalize()
            if answer == "Yes":
                break
        except:
            pass
        temp = imageQ_prompt.replace("{question}", question['question'])
        temp = temp.replace("{GraphML}", graph_to_str(essential_graph))
        response = text_request(temp, API_URL)
        images_question_list = extract_answer_list(response)
        try:
            if len(images_question_list) != 0:
                for item in images_question_list:
                    for q_image_entity_name, image_question in item.items():
                        for node in essential_graph.nodes(data=True):
                            if node[1]['entity_name'].strip('"') == q_image_entity_name.strip('"') and node[1][
                                'type'] == "IMAGE":

                                number_api['image_k'] += 1
                                result = get_answer(
                                    node[1]['entity_name'],
                                    os.path.join(
                                        mmqa_images_dir(),
                                        image_data_dict[node[1]['source_id']][0]['path'],
                                    ),
                                    image_question,
                                )
                                if result is not None and result != "":
                                    node[1]['description'] += image_question + "\nAnswer: " + result
                                    break
        except Exception as e:
            print(e)
        for node in list(essential_graph.nodes):
            if G.has_node(node):
                neighbors = list(G.neighbors(node))
                for neighbor in neighbors:
                    if (not essential_graph.has_node(neighbor) and
                            nodes[node]['entity_name'] not in irrelation_graph):
                        essential_graph.add_node(neighbor, **nodes[neighbor])
                    if (essential_graph.has_node(node) and
                            essential_graph.has_node(neighbor) and
                            not essential_graph.has_edge(node, neighbor)):
                        essential_graph.add_edge(node, neighbor, **G.edges[node, neighbor])
        for node in essential_graph.nodes(data=True):
            if node[1]['type'] == "TABLE":
                temp = table_qa_prompt.replace("{Question}", question['question'])
                temp = temp.replace("{Table name}", node[1]['entity_name'])
                temp = temp.replace("{Table content}", node[1]['description'])
                response = text_request(temp, API_URL, temperature=0)

                number_api['table_k'] += 1
                additional_entities = extract_answer_list(response)
                if len(additional_entities) != 0:
                    query_embeddings = []
                    for item in additional_entities:
                        query_embedding = get_embeding(item)
                        query_embeddings.append(query_embedding)
                    results = entity_collection.query(
                        query_embeddings=query_embeddings,
                        n_results=1
                    )
                    results = list(set([item for sublist in results['documents'] for item in sublist]))
                    for entity in results:
                        if not essential_graph.has_node(entity):
                            essential_graph.add_node(entity, **nodes[entity])

                break
        k += 1
    if k == 5:
        temp = llm_answer_prompt.replace("{question}",
                                         question['question'])
        temp = temp.replace("{GraphML}", graph_to_str(essential_graph))
        response = text_request(temp, API_URL)
    else:
        temp = GET_FINAL_ANSWER_PROMPT.replace("{question}",question['question']).replace("{my_answer}", re.sub(r'^\s*Answer:\s*(Yes|No)\s*$', '', result, flags=re.MULTILINE))

        response = text_request(temp, API_URL)
    number_api['k'] = k
    final_answer = response
    detail_path = os.path.join(
        os.path.dirname(OUTPUT_JSON) or str(_BASE_DIR / "result" / _RUN_ID),
        f"{qid}_phase5_detail.json",
    )
    os.makedirs(os.path.dirname(detail_path), exist_ok=True)
    with open(detail_path, "w", encoding="utf-8") as file:
        answer = {
            "qid": qid,
            "answer": final_answer,
            "k": number_api,
            "graph": graph_to_graphml_str(essential_graph),
        }
        file.write(json.dumps(answer, ensure_ascii=False, indent=2))

def load_jsonl_data(path):
    with open(path, "r", encoding='UTF-8') as file:
        return [json.loads(line) for line in file]


if __name__ == "__main__":
    setup_logger()
    questions = load_jsonl_data(QUESTION_FILE)
    if MAX_QUESTIONS > 0:
        questions = questions[:MAX_QUESTIONS]
    predictions = {}
    retrieval_predictions = {}
    retrieval_output_json = (
        INFERENCE_RETRIEVAL_JSON.strip()
        if INFERENCE_RETRIEVAL_JSON.strip()
        else _default_retrieval_json_path(OUTPUT_JSON)
    )
    if DRY_RUN:
        for question in tqdm(questions, desc="Processing Questions (dry-run)"):
            qid = question["qid"]
            graph_path = os.path.join(GRAPH_DIR, f"{qid}_graph.graphml")
            if not os.path.exists(graph_path):
                predictions[qid] = "unknown"
                retrieval_predictions[qid] = []
                continue
            try:
                g = nx.read_graphml(graph_path)
                retrieval_predictions[qid] = extract_ranked_source_ids_from_graph(g, top_k=10)
                table_nodes = [n for n, d in g.nodes(data=True) if d.get("type") == "TABLE"]
                if table_nodes:
                    answer = g.nodes[table_nodes[0]].get("entity_name", "unknown")
                else:
                    names = [d.get("entity_name", "") for _, d in g.nodes(data=True) if d.get("entity_name")]
                    answer = names[0] if names else "unknown"
                predictions[qid] = str(answer)
            except Exception:
                predictions[qid] = "unknown"
                retrieval_predictions[qid] = []
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        with open(retrieval_output_json, "w", encoding="utf-8") as f:
            json.dump(retrieval_predictions, f, ensure_ascii=False, indent=2)
    else:
        for question in tqdm(questions, desc="Processing Questions (real)"):
            qid = question["qid"]
            graph_path = os.path.join(GRAPH_DIR, f"{qid}_graph.graphml")
            if not os.path.exists(graph_path):
                predictions[qid] = "unknown"
                retrieval_predictions[qid] = []
                continue
            try:
                g = nx.read_graphml(graph_path)
                retrieval_predictions[qid] = extract_ranked_source_ids_from_graph(g, top_k=10)
                prompt = LLM_ANSWER_PROMPT.replace("{question}", question["question"]).replace(
                    "{GraphML}", graph_to_str(g)
                )
                predictions[qid] = text_request(prompt, API_URL).strip()
            except Exception:
                predictions[qid] = "unknown"
                retrieval_predictions[qid] = []
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        with open(retrieval_output_json, "w", encoding="utf-8") as f:
            json.dump(retrieval_predictions, f, ensure_ascii=False, indent=2)
