import json
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from datetime import datetime
from dataclasses import asdict
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import sys
import os
import argparse
import asyncio
import logging
from typing import List, Dict
import numpy as np
import re
from artrag.prompt_art import PROMPTS
from artrag.base import QueryParam
from artrag.prunning import dual_passage_rerank
from artrag.llm import gpt_4o_mini_complete, gpt_4o_complete
from artrag import LightRAG, QueryParam, clip_score
import tqdm
import pdb

from inference_eval import evaluate_descriptions_semart
from inference_eval_artpedia import evaluate_descriptions_artpedia

from artrag.storage import (
    NetworkXStorage,
)
from io import StringIO
import csv


logger = logging.getLogger(__name__)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Function to load local images
def load_image(image_path):
    return Image.open(image_path)

def parse_retrieved_context(retrieved_context):
    """
    Parses retrieved context containing entities in CSV format.
    Extracts entity data and checks if id and rank are integers.
    Handles missing or malformed data gracefully.

    :param retrieved_context: String containing the context with embedded CSV data.
    :return: A list of parsed entities (nodes) and a list of errors.
    """
    # Extract Entities CSV section
    entities_section = re.search(r'-----Entities-----\s*```csv\s*(.*?)\s*```', retrieved_context, re.DOTALL)

    nodes = []
    errors = []

    if entities_section:
        entities_csv = entities_section.group(1).strip()

        # Normalize CSV format by replacing tab characters with commas
        entities_csv = entities_csv.replace(',\t', ',').replace('\t', ',')
        entities_csv=entities_csv.replace('\n\n',' ')
        # Read CSV content using DictReader
        entities_reader = csv.DictReader(StringIO(entities_csv), delimiter=',')

        for row in entities_reader:
            try:
                # Ensure 'id' is an integer
                row_id = int(row['id'].strip()) if row.get('id') and row['id'].strip().isdigit() else None
                if row_id is None:
                    errors.append(f"Invalid or missing ID: {row.get('id', 'None')}")
                    continue  # Skip this row if ID is missing or invalid
            except (ValueError, TypeError):
                errors.append(f"Invalid ID value: {row.get('id', 'None')}")
                continue  # Skip this row if ID cannot be converted

            # Handle missing or invalid rank
            rank_value = row.get('rank', '0')  # Default to '0' if missing
            try:
                rank = int(rank_value.strip()) if rank_value and rank_value.strip().isdigit() else 0
            except (ValueError, TypeError):
                errors.append(f"Invalid Rank value for entity {row.get('entity', 'Unknown')}: {rank_value}")
                rank = 0

            # Add entity details to nodes list
            nodes.append({
                'id': row_id,
                'entity_name': row.get('entity', 'Unknown').strip() if row.get('entity') else 'Unknown',
                'type': row.get('type', 'Unknown').strip() if row.get('type') else 'Unknown',
                'description': row.get('description', '').strip() if row.get('description') else '',
                'rank': rank
            })

    return nodes, errors

async def main(args, llm_model_func):

    rag = LightRAG(
        working_dir = args.WORKING_DIR,
        llm_model_func = llm_model_func  # Use the specified LLM model function
    )
    knowledge_graph_inst = NetworkXStorage(
        namespace="chunk_entity_relation", global_config=asdict(rag)
    )

    json_file_path = args.json_file_path
    data_type = args.data_type
    print("model name: ", args.model_name)
    if args.model_name == "Qwen2-VL":
        model_name = "Qwen/Qwen2-VL-7B-Instruct"
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)

    if args.model_name == "Qwen2_5-VL":
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)

    with open(json_file_path, 'r') as f:
        paintings = json.load(f)

    results = []
    for painting in tqdm.tqdm(paintings):
        title = painting.get("Title", "Unknown Title")
        author = painting.get("Author", "Unknown Author")
        concepts = painting.get("Concepts", "")
        Timeframe = painting.get("Timeframe", "")
        retrieved_context = painting.get("Retrieved context", "")
        image = painting.get("Image", "")

        if data_type == "SemArtv2":
            image_path = os.path.join('../data/SemArt/Images', image)
            prompt = PROMPTS["rag_SemArtv2_2-shot_incontext_response"]
            # prompt = PROMPTS["rag_SemArtv2_2-shot_incontext_response_v2"]
        elif data_type == "Artpedia":   
            image_path = os.path.join('../data/Artpedia/Images', str(image)+".jpg")
            prompt = PROMPTS["rag_SemArtv2_2-shot_incontext_response"]
            # prompt = PROMPTS["rag_SemArtv2_2-shot_incontext_response_v2"]
        
        try:
            images = load_image(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        entities,_ = parse_retrieved_context(retrieved_context)
        # import pdb; pdb.set_trace()

        painting_metadata = {
            "title": title,
            "author": author,
            "concepts": concepts,
            "Timeframe": Timeframe
        }
        painting_metadata = "Metadata: " + json.dumps(painting_metadata)

        reranked_nodes, inter_edges = await dual_passage_rerank(image_path, painting_metadata, entities, llm_model_func, knowledge_graph_inst)
        logger.info(f"Reranked nodes: {len(reranked_nodes)} and inter_edges: {len(inter_edges)}")
        # import pdb; pdb.set_trace()
        subgraph_context = "-----Entities-----\n"
        subgraph_context += "```csv\n"
        subgraph_context += "id,\tentity,\ttype,\tdescription,\trank\n"

        for node in reranked_nodes:
            subgraph_context += f"{node['id']}, {node['entity_name']}, {node['type']}, {node['description']}, {node['rank']}\n"
        subgraph_context += "-----Relationships-----\n"
        subgraph_context += "```csv\n"
        subgraph_context += "id,\tsource and target,\tdescription,\tweight\n"

        for edge in inter_edges:
            subgraph_context += f"{edge['src_tgt']}, {edge['description']}, {edge['weight']}\n"

        response_type = "Keep your generated description strictly under 15 words."
        prompt = prompt.format(response_type=response_type, context_data=subgraph_context)
        prompt += f"""
        Title: {title}
        Author: {author}
        Generate a detailed explanation for this painting. with requirment of {response_type}
        """
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("generated explanations: ", generated_text[0].split("assistant")[2])
        
        results.append({
            "Title": title,
            "Author": author,
            "Image": image,
            "Generated Description": generated_text[0].split("assistant")[2],
            "Reranked Nodes": reranked_nodes,
            "Interconnected Edges": inter_edges
        })
        print(f"Processed painting: {title}")

    output_path = os.path.join(os.path.dirname(json_file_path),  "{}_explanations_{}.json".format(args.model_name, timestamp))
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Generated descriptions saved to {output_path}")

    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process paintings and generate descriptions.")
    parser.add_argument("--json_file_path", type=str, default="./built_graph/All_gpt_4o_mini_prompt_tuning/output_2025-01-27_SemArtv2_900data/generated_descriptions_local_20250127_094917.json",
                        help="Path to the JSON file containing painting data")
    parser.add_argument("--model_name", type=str, default="Qwen2-VL",
                        choices = ["Qwen2-VL","Qwen2_5-VL" ] ,help="Name of the model used for evaluation")
    parser.add_argument("--data_type", type=str, default="SemArtv2",
                        choices = ["SemArtv2", "Artpedia" ] , help="Batch size for evaluation")
    parser.add_argument("--WORKING_DIR", type=str, default="./built_graph/All_gpt_4o_mini_prompt_tuning_style_event_clean",)
    parser.add_argument("--llm_model_func", type=str,default="gpt_4o_mini_complete",)
    
    args = parser.parse_args()

    llm_model_func_map = {
        'gpt_4o_mini_complete': gpt_4o_mini_complete,
        'gpt_4o_complete': gpt_4o_complete
    }
    output_path = asyncio.run(main(args, llm_model_func_map[args.llm_model_func]))

    data_type = args.data_type
    if data_type == "SemArtv2":
        evaluate_descriptions_semart(output_path, data_type,  args.model_name, batch_size=8)
    elif data_type == "Artpedia":
        evaluate_descriptions_artpedia(output_path, data_type,  args.model_name, batch_size=8)