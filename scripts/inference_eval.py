import clip
import torch
import ast
import logging
import pdb
from multiprocessing import Pool, cpu_count
import sys
import os
sys.path.append(os.path.abspath('.'))
import numpy as np
from pprint import pprint
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import json
import re
import argparse

from artrag.llm import gpt_4o_mini_complete, gpt_4o_complete
from artrag import LightRAG, QueryParam, clip_score
import language_evaluation



timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
current_date = datetime.now().strftime("%Y-%m-%d")


def clean_description(description):
    """
    post processing on the generated text by deleting unrevelant symbols
    """
    # Remove markdown headers and formatting
    description = re.sub(r'###\s*Description of.*\n', '', description)
    description = re.sub(r'\*\*Content\*\*:', '', description)
    description = re.sub(r'\*\*Context\*\*:', '', description)
    description = re.sub(r'\*\*Form\*\*:', '', description)

    # Remove newline characters
    description = re.sub(r'\n', ' ', description)

    # Remove double quotes
    description = re.sub(r'"', '', description)

    # Remove other unwanted characters (e.g., **)
    description = re.sub(r'\*\*', '', description)

    # Normalize whitespace
    description = re.sub(r'\s+', ' ', description)

    # Remove leading and trailing whitespace
    description = description.strip()
    return description


def run_ArtRAG_inference(WORKING_DIR, llm_model_func, args):
    """
    Run inference on the specified dataset using the specified LightRAG model.
    """

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func  # Use the specified LLM model function
    )

    semartv1_types = {"SemArtv1-content", "SemArtv1-context"}
    if args.data_type in semartv1_types:
        directory = "../data/SemArt/semartv1_test_overlap_with_captions.csv"
    elif args.data_type == "SemArtv2":
        directory = "../data/SemArt/semartv2_test_overlap_with_captions.csv"
    print("dataset type: {}, question type: {}".format(args.data_type, args.question_type))

    # Load the data from the specified directory
    data = pd.read_csv(directory, encoding='latin1', delimiter=';')[:args.data_num]

    # Placeholder for storing results
    results = []
    # Iterate through each row in the dataset
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
        print("Processing row: ", index)
        # Extract required features for each painting
        tags = row['tags']
        author = row['AUTHOR']
        img_id = row['IMAGE_FILE']
        title = row['TITLE']
        technique = row['TECHNIQUE']
        Timeframe = row['TIMEFRAME']
        img = f"../data/SemArt/Images/{row['IMAGE_FILE']}"

        semartv1_content_types = {"SemArtv1-content", "SemArtv1-context"}
        if args.data_type in semartv1_content_types:
            form_exists = False
            if args.data_type == "SemArtv1-content":
                context_exists = False
            elif args.data_type == "SemArtv1-context":
                content_exists = False
        elif args.data_type == "SemArtv2":
            form = row['form']
            form_exists = pd.notna(form) and form != '[]'

        content = row['content']
        context = row['context']

        context_exists = pd.notna(context) and context != '[]'
        content_exists = pd.notna(content) and content != '[]'

        # Create input for LightRAG model based on non-empty attributes
        if args.question_type == "description":
            input_text = f"Please generate the description on "
            if context_exists:
                input_text += " --context--,"
            if form_exists:
                input_text += " --form--,"
            if content_exists:
                input_text += " --content--,"
            input_text+="perspective of the painting"
            input_text = input_text.rstrip(',')  
        elif args.question_type == "cultural&histroical":
            input_text = f"What historical events or cultural movements influenced the creation of this painting? "
        elif args.question_type == "Theme":
            input_text = f"What themes or beliefs are reflected in this painting? "
        elif args.question_type == "style&technique":
            input_text = f"Does the painting reflect a shift from one stylistic period to another? "
        elif args.question_type == "Movement&school":
            input_text = f"How does this painting embody the principles of its art movement? "
        elif args.question_type == "artist":
            input_text = f"How does this painting reflect the artist’s personal beliefs or experiences? "
        
        # Remove trailing comma
        input_text += f"with painting Metadata:  {title}, Author: {author}, Technique: {technique}, Timeframe: {Timeframe}, Painting Concepts: {tags}"
        query = {"text": input_text, "image": img}

        # Run inference with LightRAG model
        generated_description , retrieved_context, rerank_context = rag.query(
            query, param=QueryParam(mode=args.retrieval_strategy), data_type=args.data_type, shot_number=args.shot_number, fewshot_type=args.fewshot_type)
        # Store the result
        # import pdb; pdb.set_trace()
        print("generated_description: ", generated_description)
        results.append({
            'Title': title,
            'Image': img_id,
            'Author': author,
            'Technique': technique,
            'Timeframe': Timeframe,
            "Concepts": tags,
            'Generated Description': generated_description,  # Ensure newlines are escaped,
            'Retrieved context': retrieved_context,  # Ensure newlines are escaped
            'rerank_context': rerank_context
        })

    # Convert results to a DataFrame for easier viewing or saving
    results_df = pd.DataFrame(results)

    output_DIR = os.path.join(WORKING_DIR, "output_{}_{}_{}data".format(
        current_date, args.data_type, args.data_num))
    if not os.path.exists(output_DIR):
        os.mkdir(output_DIR)

    # Save the results and args to JSON files with indentation for readability
    output_file = os.path.join(output_DIR,
                               'generated_descriptions_{}_{}.json'.format(args.retrieval_strategy, timestamp))
    results_df.to_json(output_file, orient='records',
                       indent=4, force_ascii=False)
    
    # Save args as JSON
    args_dict = vars(args)
    args_file = os.path.join(output_DIR,
                            'args_{}.json'.format(timestamp))
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    print(
        f"Inference completed. Generated descriptions saved to '{output_file}'.")

    return output_file


def evaluate_batch(batch_predicts, batch_answers):
    # evaluator = language_evaluation.CocoEvaluator()
    # return evaluator.run_evaluation(batch_predicts, batch_answers)
    # pdb.set_trace()
    try:
        evaluator = language_evaluation.CocoEvaluator()
        batch_result = evaluator.run_evaluation(batch_predicts, batch_answers)
        return batch_result
    except Exception as e:
        print(f"Error evaluating batch: {e}")
        return None


def evaluate_descriptions_semart(generated_descriptions_file, data_type, model_name, batch_size=8):
    # Load generated descriptions
    with open(generated_descriptions_file, 'r') as f:
        generated_descriptions = json.load(f)

    semartv1_types = {"SemArtv1-content", "SemArtv1-context"}
    if data_type in semartv1_types:
        ground_truth_file = "../data/SemArt/semartv1_test_overlap_with_captions.csv"
    elif data_type == "SemArtv2":
        ground_truth_file = "../data/SemArt/semartv2_test_overlap_with_captions.csv"

    # Load ground truth descriptions
    ground_truth_descriptions = {}
    df = pd.read_csv(ground_truth_file, encoding='latin1', delimiter=';')

    for index, row in df.iterrows():
        content = ast.literal_eval(row['content'])
        form = ast.literal_eval(row['form']) if 'form' in row else []
        context = ast.literal_eval(row['context'])

        if data_type == "SemArtv1-content":
            full_description = content
            image_dir = "../data/SemArt/Images"
        elif data_type == "SemArtv1-context":
            full_description = context
            image_dir = "../data/SemArt/Images"
        elif data_type == "SemArtv2":
            full_description = content + form + context
            image_dir = "../data/SemArt/Images"
        ground_truth_descriptions[row['IMAGE_FILE']] = full_description
    # Prepare lists for evaluation
    predicts = []
    answers = []

    for description in generated_descriptions:
        image_id = description["Image"]
        if image_id in ground_truth_descriptions and ground_truth_descriptions[image_id] != []:
            cleaned_description = clean_description(
                description['Generated Description'])
            predicts.append(cleaned_description)
            answers.append(ground_truth_descriptions[image_id])
        else:
            print(f"Ground truth description not found for image: {image_id}")
    # import pdb; pdb.set_trace()
    # Calculate the average word length of answers
    total_words = sum(len(predict.split()) for predict in predicts)
    average_word_length = total_words / len(answers) if answers else 0

    print("Average word length of answers: {:.2f}".format(average_word_length))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device, jit=False)

    model.eval()
    image_ids = [i["Image"] for i in generated_descriptions]
    image_paths = [os.path.join(image_dir, path) for path in image_ids]
    # import pdb; pdb.set_trace()
    image_feats = clip_score.extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8)

    _, per_instance_image_text, candidate_feats = clip_score.get_clip_score(
        model, image_feats, [i['Generated Description'] for i in generated_descriptions], device)

    scores = {image_id: {'CLIPScore': float(clipscore)}
              for image_id, clipscore in
              zip(image_ids, per_instance_image_text)}
    print('CLIPScore: {:.4f}'.format(
        np.mean([s['CLIPScore'] for s in scores.values()])))

    # Evaluate using language_evaluation tool in batches
    num_samples = len(predicts)
    # Calculate the number of batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    all_scores = []
    mean_scores = {}
    output_dir = os.path.dirname(generated_descriptions_file)
    mean_scores_file = os.path.join(
        output_dir, 'mean_scores_{}_{}.json'.format(model_name, timestamp))

    for i in tqdm(range(num_batches), desc="Evaluating batches"):
        batch_predicts = predicts[i * batch_size:(i + 1) * batch_size]
        batch_answers = answers[i * batch_size:(i + 1) * batch_size]
        batch_result = evaluate_batch(batch_predicts, batch_answers)
        if batch_result is not None:
            all_scores.append(batch_result)
            for metric in batch_result.keys():
                if metric not in mean_scores:
                    mean_scores[metric] = []
                mean_scores[metric].append(batch_result[metric])
            pprint(batch_result)
        else:
            print("batch_result is None")

    # Calculate mean scores for each metric
    for metric in mean_scores.keys():
        mean_scores[metric] = np.mean(mean_scores[metric])
    print("Mean coco Scores:", mean_scores)

    # Save mean scores to the same folder as generated_descriptions_file
    with open(mean_scores_file, 'w') as f:
        json.dump(mean_scores, f, indent=4)

    print(f"All Mean scores saved to '{mean_scores_file}'.")


if __name__ == "__main__":
    # Set up argument parser for inference and evaluation
    parser = argparse.ArgumentParser(
        description="Run LightRAG inference and evaluate generated descriptions.")
    
    # Model and data configuration
    parser.add_argument(
        '--working_dir', 
        type=str,
        default="./built_graph/All_gpt_4o_mini_prompt_tuning_style_event_clean",
        help='Working directory for LightRAG.'
    )
    parser.add_argument(
        '--llm_model_func',
        type=str, 
        default='gpt_4o_mini_complete',
        choices=['gpt_4o_mini_complete', 'gpt_4o_complete'],
        help='LLM model function to use.'
    )
    parser.add_argument(
        '--data_type',
        type=str,
        default="SemArtv2",
        choices=["SemArtv1-content", "SemArtv1-context", "SemArtv2"],
        help='Dataset type to use for inference.'
    )
    
    # Inference settings
    parser.add_argument(
        '--shot_number',
        type=int,
        default=1,
        choices=[0, 1, 2, 3],
        help='Number of shots for in-context learning'
    )
    parser.add_argument(
        '--retrieval_strategy',
        type=str,
        default="local",
        choices=['local', 'global', 'hybrid', 'naive', 'no-rag'],
        help='Retrieval strategy to use.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for evaluation.'
    )
    parser.add_argument(
        '--data_num',
        type=int,
        default=100,
        help='Number of data samples to process'
    )
    
    # Input/Output settings
    parser.add_argument(
        '--generated_descriptions',
        type=str,
        help='Pre-generated caption file path (if exists, inference will be skipped)'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default='../data/SemArt/Images',
        help='Directory of images, with the filenames as image ids.'
    )
    
    # Processing options
    parser.add_argument(
        '--mp',
        action='store_true',
        help='Enable multiprocessing (currently not functional)'
    )
    parser.add_argument(
        '--fewshot_type',
        type=str,
        default="SM_fewshot",
        choices=['MM_fewshot', 'SM_fewshot'],
        help='Type of few-shot learning examples: MultiModal (MM) or SingleModal (SM)'
    )

    parser.add_argument(
        '--question_type',
        type=str,
        default="description",
        choices=["description", "cultural&histroical", "Theme", "style&technique","Movement&school", "artist"],
        help='Type of question to generate: description or question'
    )


    args = parser.parse_args()
    print("args: ", args)

    # Map the llm_model_func argument to the actual function
    llm_model_func_map = {
        'gpt_4o_mini_complete': gpt_4o_mini_complete,
        'gpt_4o_complete': gpt_4o_complete
    }
    if not args.generated_descriptions:
        # Run inference
        generated_descriptions_file = run_ArtRAG_inference(
            args.working_dir, llm_model_func_map[args.llm_model_func], args)
    else:
        generated_descriptions_file = args.generated_descriptions

    print("args.batch_size :", args.batch_size)
    # Evaluate descriptions
    evaluate_descriptions_semart(generated_descriptions_file, args.data_type,
                          args.llm_model_func, args.batch_size)
