import pandas as pd 
import os
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from pydantic import BaseModel, Field
from typing import List, Optional
from typing import List, Literal
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from dotenv import load_dotenv
load_dotenv()
from datetime import date
from google.genai import types
from pydantic import ValidationError
import re,logging,time
from src.data_pipelines.survey.prompts import PainPointAnalyzer, MODAnalyzer
from src.prompts.loader import get_survey_subtopic_map_prompt
from src.core.llm import make_model
log = logging.getLogger(__name__)

def load_batches_subtopics(output_dir):
    """
    Reads CSVs named 'review_batch_<n>.csv' from output_dir up to stop_batch (inclusive)
    and concatenates them into a single DataFrame.
    """
    dfs = []
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"Loaded batch {file} ({len(df)} rows)")
        else:
            print(f"⚠️ File not found: {file_path}")
            break  # stop if a batch file is missing (optional)

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\n✅ Combined {len(dfs)} batches into a single DataFrame ({len(combined_df)} rows).")
        return list(
            pd.Series(combined_df['subtopic'].dropna().tolist())
            .drop_duplicates()
            .sort_values()
        )
    else:
        print("❌ No batches loaded.")
        return []
    
# def pp_mod_topic_categorization(data_list, batch_size, model, output_folder, mode, subtopics_list=[]):
    
#     # Ensure the output folder exists
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     df_list = []
#     # subtopics_list = []
#     # Keep track of completed batches
#     completed_batches = set(int(file.split('_')[-1].split('.')[0]) for file in os.listdir(output_folder) if file.endswith('.csv'))

#     # Split the data into batches
#     total_batches = (len(data_list) + batch_size - 1) // batch_size
#     for batch_num in range(total_batches):
#         if batch_num in completed_batches:
#             print(f"Skipping already processed batch {batch_num}")
#             continue

#         # Get the current batch
#         start_index = batch_num * batch_size
#         end_index = start_index + batch_size
#         data_list_sample = data_list[start_index:end_index]

#         if not data_list_sample:
#             break  # No more data to process
        
        
#         if mode == "pain_point":
#             Analyzer = PainPointAnalyzer
#             obj_label = "pain point"
#             obj_label_plural = "pain points"
#             representatation = "Pain Point"
#             pos_neg = "negative"
#             key_col   = "pain_point"
#         else:
#             Analyzer = MODAnalyzer
#             obj_label = "moment of delight"
#             obj_label_plural = "moments of delight"
#             representatation = "Delight"
#             pos_neg = "positive"
#             key_col   = "mod"
#         # Set the system content and Analyzer class based on the mode
        
        
#         ppm_cfg = generative_models.GenerationConfig(
#             response_mime_type="application/json",
#             response_schema=Analyzer.model_json_schema(),
#             temperature=0,
#             max_output_tokens=60000
#         )
#         if subtopics_list == []:        
#             batch_message = [get_initial_pp_mod_sys_prompt(obj_label_plural,obj_label,pos_neg)] + [m["content"] for m in data_list_sample]
#         else:
#             batch_message = [get_second_pp_mod_sys_prompt(obj_label_plural,obj_label,pos_neg,subtopics_list)] + [m["content"] for m in data_list_sample]
        
#         resp   = model.generate_content(batch_message, generation_config=ppm_cfg)
#         result = Analyzer.model_validate_json(resp.text)

#         # Transform the result into a DataFrame
#         data = [
#             {key_col: review.__dict__[key_col], 'topic':review.topic ,'subtopic': review.subtopic}
#             for review in result.reviews
#         ]
#         batch_data = pd.DataFrame(data)
        
#         subtopics_list = list(
#             pd.Series(subtopics_list + batch_data['subtopic'].dropna().tolist())
#             .drop_duplicates()
#             .sort_values()
#         )

        
#         df_list.append(batch_data)
#         # Save the batch results to a CSV file
#         output_file = os.path.join(output_folder, f'batch_{batch_num}.csv')
#         batch_data.to_csv(output_file, index=False)
#         print(f"Saved batch {batch_num} to {output_file}")
    
#     if df_list:
#         print("All batches combined and saved.")
#     else:
#         print("No new batches were processed.")

def pp_mod_topic_categorization_standard(data_list, batch_size, model, output_folder, mode):
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df_list = []
    # subtopics_list = []
    # Keep track of completed batches
    completed_batches = set(int(file.split('_')[-1].split('.')[0]) for file in os.listdir(output_folder) if file.endswith('.csv'))

    # Split the data into batches
    total_batches = (len(data_list) + batch_size - 1) // batch_size
    for batch_num in range(total_batches):
        if batch_num in completed_batches:
            print(f"Skipping already processed batch {batch_num}")
            continue

        # Get the current batch
        start_index = batch_num * batch_size
        end_index = start_index + batch_size
        data_list_sample = data_list[start_index:end_index]

        if not data_list_sample:
            break  # No more data to process
        
        
        if mode == "pain_point":
            Analyzer = PainPointAnalyzer
            obj_label = "pain point"
            obj_label_plural = "pain points"
            representatation = "Pain Point"
            pos_neg = "negative"
            key_col   = "pain_point"
        else:
            Analyzer = MODAnalyzer
            obj_label = "moment of delight"
            obj_label_plural = "moments of delight"
            representatation = "Delight"
            pos_neg = "positive"
            key_col   = "mod"
        # Set the system content and Analyzer class based on the mode
        
        
        ppm_cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Analyzer.model_json_schema(),
            temperature=0,
            max_output_tokens=60000
        )
                
        batch_message = [get_survey_subtopic_map_prompt(obj_label_plural, obj_label)] + [m["content"] for m in data_list_sample]
        
        resp   = model.generate_content(batch_message, generation_config=ppm_cfg)
        result = Analyzer.model_validate_json(resp.text)

        # Transform the result into a DataFrame
        data = [
            {key_col: review.__dict__[key_col], 'topic':review.topic ,'subtopic': review.subtopic}
            for review in result.reviews
        ]
        batch_data = pd.DataFrame(data)

        
        df_list.append(batch_data)
        # Save the batch results to a CSV file
        output_file = os.path.join(output_folder, f'batch_{batch_num}.csv')
        batch_data.to_csv(output_file, index=False)
        print(f"Saved batch {batch_num} to {output_file}")
    
    if df_list:
        print("All batches combined and saved.")
    else:
        print("No new batches were processed.")

def separate_pp_mod(df):
    # Process pain points
    pain_points_combined = df[['id', 'sentiment', 'topic', 'pain_points']].dropna()
    pain_points_combined['pain_points'] = pain_points_combined['pain_points'].str.split(',')
    pain_points_combined = pain_points_combined.explode('pain_points')
    pain_points_combined['pain_points'] = pain_points_combined['pain_points'].str.lower().str.strip()
    
    # Rename column in pain_points_combined
    pain_points_combined = pain_points_combined.rename(columns={'id': 'Review_No'})
    # pain_points_combined['pp_mod_id']= pain_points_combined['Review_No'].astype(str) + "-" + pain_points_combined.index.astype(str)
    # pain_points_combined.to_csv('pain_points_combined.csv', index=False, encoding='utf-8-sig')
    # Process moments of delight
    mod_combined = df[['id', 'sentiment', 'topic', 'moments_of_delight']].dropna()
    mod_combined['moments_of_delight'] = mod_combined['moments_of_delight'].str.split(',')
    mod_combined = mod_combined.explode('moments_of_delight')
    mod_combined['moments_of_delight'] = mod_combined['moments_of_delight'].str.lower().str.strip()
    
    # Rename column in mod_combined
    mod_combined = mod_combined.rename(columns={'id': 'Review_No'})
    # mod_combined['pp_mod_id']= mod_combined['Review_No'].astype(str) + "-" + mod_combined.index.astype(str)
    # mod_combined.to_csv('mod_combined.csv', index=False, encoding='utf-8-sig')
    return pain_points_combined, mod_combined

def process_topics_subtopics(df, model):
    pain_points, mods = separate_pp_mod(df)
    
    pp_unique = pain_points[['pain_points', 'topic']].drop_duplicates()
    mod_unique = mods[['moments_of_delight', 'topic']].drop_duplicates()

    pp_messages = [
        {"role": "user",
         "content": (
            f"Pain_Point: {row.pain_points}\n"
            f"Current_Topic: {row.topic}"
        )}
        for _, row in pp_unique.iterrows()
    ]

    mod_messages = [
        {"role": "user",
         "content": (
            f"Moment_of_Delight: {row.moments_of_delight}\n"
            f"Current_Topic: {row.topic}"
        )}
        for _, row in mod_unique.iterrows()
    ]
    pp_mod_topic_categorization_standard(data_list=pp_messages, batch_size=150, model=model, output_folder=f'./output/new_subtopics/painpoints_topics/{date.today()}_batches',
                                  mode='pain_point')
    pp_mod_topic_categorization_standard( data_list=mod_messages, batch_size=150, model=model, output_folder=f'./output/new_subtopics/mod_topics/{date.today()}_batches', 
                                mode='mod')


    # pp_mod_topic_categorization(data_list=pp_messages, batch_size=150, model=model, output_folder=f'./output/painpoints_topics/{date.today()}_batches',
                                #   mode='pain_point')
    
    # subtopics = load_batches_subtopics(f'./output/painpoints_topics/{date.today()}_batches')
    # pp_mod_topic_categorization( data_list=mod_messages, batch_size=150, model=model, output_folder=f'./output/mod_topics/{date.today()}_batches', 
    #                             mode='mod', subtopics_list=subtopics)

if __name__ == "__main__":
    model = make_model(os.environ.get('GEMINI_API_KEY'))
    df = pd.read_csv('./output/2025-10-27_batches/all_batches_combined_mapped_topics.csv')
    process_topics_subtopics(df, model)