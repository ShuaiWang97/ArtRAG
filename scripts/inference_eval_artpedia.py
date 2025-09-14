import os
import sys
sys.path.append(os.path.abspath('.'))
import argparse
import re
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from pprint import pprint
import numpy as np
import language_evaluation
from artrag import LightRAG, QueryParam, clip_score
from artrag.llm import gpt_4o_mini_complete, gpt_4o_complete
from multiprocessing import Pool, cpu_count
import pdb
import logging
import ast
import torch
import clip

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


def run_inference(WORKING_DIR, llm_model_func, data_type, retrieval_strategy, shot_number, data_num=500):

    """
    Run inference on the specified dataset using the specified LightRAG model.
    """

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func  # Use the specified LLM model function
    )
    # import pdb; pdb.set_trace()
    if data_type == "Artpedia":
        directory = "../data/Artpedia/artpedia_test.json"
        data = pd.read_json(directory, encoding='utf8',orient = "index")[:data_num]
    elif data_type == "ExpArt":
        directory = "../data/ExpArt/test_unseen-artist-year.json"
        data = pd.read_json(directory, encoding='utf8')[:data_num]



    # Placeholder for storing results
    results = []
    # Iterate through each row in the dataset
    for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
        print("Processing row: ", index)
        # Extract required features for each painting
        if data_type == "Artpedia":
            tags = row['tags']
            artist = row['artists']
            img_id = row.name
            img = f"../data/Artpedia/Images/{img_id}.jpg"
        elif data_type == "ExpArt":
            tags = row['entities']
            artist = row['artist']
            img_id = row['ID']
            img = f"../data/ExpArt/Images/{img_id[:4]}.jpg"
            if not os.path.exists(img):
                print(f"Image {img} does not exist. Skipping.")
                continue
        
        title = row['title']
        year = row['year']
        

        # Define the presence of each attribute
        form_exists = False
        context_exists = True
        content_exists = True

        # Create input for LightRAG model based on non-empty attributes
        input_text = f"Please generate the description on "
        if context_exists:
            input_text += " --context--,"
        if form_exists:
            input_text += " --form--,"
        if content_exists:
            input_text += " --content--,"
        # input_text = input_text.rstrip(',')  # Remove trailing comma
        input_text += f" perspective of the painting with Metadata:  {title},  Year: {year}, simple description: {tags}"
        query = {"text": input_text, "image": img}

        # Run inference with LightRAG model
        generated_description,retrieved_context,rerank_context = rag.query(
            query, param=QueryParam(mode=retrieval_strategy),data_type=data_type, shot_number=shot_number)
        # Store the result
        print("generated_description: ", generated_description)
        results.append({
            'Title': title,
            'Image': img_id,
            'Author': artist,
            "Concepts": tags,
            'Year': year,
            'Retrieved context': retrieved_context,
            'Generated Description': generated_description,  # Ensure newlines are escaped
            'rerank_context': rerank_context
            
        })

    # Convert results to a DataFrame for easier viewing or saving
    results_df = pd.DataFrame(results)

    output_DIR = os.path.join(WORKING_DIR, "output_{}_{}_{}_{}_data".format(current_date, data_type, retrieval_strategy,data_num))
    if not os.path.exists(output_DIR):
        os.mkdir(output_DIR)

    # Save the results to a JSON file with indentation for readability
    output_file = os.path.join(output_DIR,
                               'generated_descriptions_{}_{}.json'.format(retrieval_strategy, timestamp))
    results_df.to_json(output_file, orient='records',
                       indent=4, force_ascii=False)

    print(
        f"Inference completed. Generated descriptions saved to '{output_file}'.")

    return output_file


def evaluate_batch(batch_predicts, batch_answers):
    # evaluator = language_evaluation.CocoEvaluator()
    # return evaluator.run_evaluation(batch_predicts, batch_answers)
    # pdb.set_trace()
    try:
        evaluator = language_evaluation.CocoEvaluator()
        batch_result  = evaluator.run_evaluation(batch_predicts, batch_answers)
        return batch_result
    except Exception as e:
        print(f"Error evaluating batch: {e}")
        return None


def evaluate_descriptions_artpedia(generated_descriptions_file, data_type, model_name, batch_size=8, mp=False):
    # Load generated descriptions
    with open(generated_descriptions_file, 'r') as f:
        generated_descriptions = json.load(f)
    
    if data_type == "Artpedia":
        ground_truth_file = "../data/Artpedia/artpedia_test.json"
        df = pd.read_json(ground_truth_file, encoding='utf8',orient = "index")
        image_dir = "../data/Artpedia/Images"
    elif data_type == "ExpArt":
        ground_truth_file = "../data/ExpArt/test_unseen-artist-year.json"
        image_dir = "../data/ExpArt/Images"
        df = pd.read_json(ground_truth_file, encoding='utf8')
    

    
    # Load ground truth descriptions
    ground_truth_descriptions = {}
    for index, row in df.iterrows():

        if data_type == "Artpedia":
            content = row['contextual_sentences']
            context = row['visual_sentences']

            full_description = content + context
            ground_truth_descriptions[row.name] = full_description
        elif data_type == "ExpArt":
            full_description = row['answer']
            ground_truth_descriptions[row.ID] = full_description
        
    
    # Prepare lists for evaluation
    predicts = []
    answers = []
    # import pdb; pdb.set_trace()
    for description in generated_descriptions:
        image_id = description["Image"]
        if image_id in ground_truth_descriptions:
            cleaned_description = clean_description(
                description['Generated Description'])
            predicts.append(cleaned_description)
            answers.append(ground_truth_descriptions[image_id])
        else:
            print(f"Ground truth description not found for image: {image_id}")

    # Calculate the average word length of answers
    total_words = sum(len(predict.split()) for predict in predicts)
    average_word_length = total_words / len(answers) if answers else 0

    print("Average word length of answers: {:.2f}".format(average_word_length))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device, jit=False)

    model.eval()

    if data_type == "Artpedia":
        image_ids = [str(i["Image"])+".jpg" for i in generated_descriptions]    
    elif data_type == "ExpArt":
        image_ids = [str(i["Image"][:4])+".jpg" for i in generated_descriptions]
    image_paths = [os.path.join(image_dir, path) for path in image_ids]

    image_feats = clip_score.extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8)

    _, per_instance_image_text, candidate_feats = clip_score.get_clip_score(
        model, image_feats, predicts, device)
    

    scores = {image_id: {'CLIPScore': float(clipscore)}
                for image_id, clipscore in
                zip(image_ids, per_instance_image_text)}
    print('CLIPScore: {:.4f}'.format(np.mean([s['CLIPScore'] for s in scores.values()])))

    # import pdb; pdb.set_trace()
    # Evaluate using language_evaluation tool in batches
    num_samples = len(predicts)
    # Calculate the number of batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    all_scores = []
    mean_scores = {}
    output_dir = os.path.dirname(generated_descriptions_file)
    mean_scores_file = os.path.join(
        output_dir, 'mean_scores_{}_{}_{}.json'.format(model_name, data_type, timestamp))

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
        else :
            print("batch_result is None")

    # Calculate mean scores for each metric
    for metric in mean_scores.keys():
        mean_scores[metric] = np.mean(mean_scores[metric])
    print("Mean coco Scores:",mean_scores)

    # Save mean scores to the same folder as generated_descriptions_file
    with open(mean_scores_file, 'w') as f:
        json.dump(mean_scores, f, indent=4)

    print(f"All Mean scores saved to '{mean_scores_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LightRAG inference and evaluate generated descriptions.")
    parser.add_argument('--working_dir', default="./built_graph/All_gpt_4o_mini_prompt_tuning_style_event_clean",type=str,
                        help='Working directory for LightRAG.')
    
    parser.add_argument('--llm_model_func', default='gpt_4o_mini_complete', type=str,
                        choices=['gpt_4o_mini_complete', 'gpt_4o_complete'], help='LLM model function to use.')
    
    parser.add_argument('--data_type', type=str, default="Artpedia", choices=["Artpedia","ExpArt","SemArtv1-content", "SemArtv1-context","SemArtv2"],
                        help='Dataset type to use for inference.')
    
    parser.add_argument('--shot_number',type=int, default=2, choices=[1,2, 3],
                        help='Number of shots for in-context learning')
    
    parser.add_argument('--retrieval_strategy', type=str, default="local", choices=[
                        'local', 'global', 'hybrid', 'naive', 'no-rag'], help='Retrieval strategy to use.')
    
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation.')
    
    parser.add_argument('--generated_caption', type=str,
                        help='if there is pre_generated_caption file, then skip the inference')
    
    parser.add_argument('--mp',  action='store_true',
                        help='if using multiprocessing. Not working yet')
    parser.add_argument('--data_num', type=int, default=100,)

    parser.add_argument(
        '--image_dir',
        type=str,
        default='../data/Artpedia/Images',
        help='Directory of images, with the filenames as image ids.')

    args = parser.parse_args()
    print("args: ", args)

    # Map the llm_model_func argument to the actual function
    llm_model_func_map = {
        'gpt_4o_mini_complete': gpt_4o_mini_complete,
        'gpt_4o_complete': gpt_4o_complete
    }
    if not args.generated_caption:
        # Run inference
        generated_descriptions_file = run_inference(
            args.working_dir, llm_model_func_map[args.llm_model_func], args.data_type, args.retrieval_strategy, args.shot_number, args.data_num)
    else:
        generated_descriptions_file = args.generated_caption

    print("args.batch_size :", args.batch_size)
    # Evaluate descriptions
    evaluate_descriptions_artpedia(generated_descriptions_file, args.data_type, model_name=args.llm_model_func, batch_size= args.batch_size)
