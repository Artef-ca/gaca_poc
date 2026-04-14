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
from src.data_pipelines.survey.prompts import ReviewsAnalyzerGem
from src.prompts.loader import get_survey_first_pass_prompt, get_survey_second_pass_prompt
from src.core.llm import make_model
log = logging.getLogger(__name__)
def load_batches_topics(output_dir, stop_batch):
    """
    Reads CSVs named 'review_batch_<n>.csv' from output_dir up to stop_batch (inclusive)
    and concatenates them into a single DataFrame.
    """
    dfs = []
    for batch_number in range(1, stop_batch):
        file_path = os.path.join(output_dir, f'review_batch_{batch_number}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)
            print(f"Loaded batch {batch_number} ({len(df)} rows)")
        else:
            print(f"⚠️ File not found: {file_path}")
            break  # stop if a batch file is missing (optional)

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\n✅ Combined {len(dfs)} batches into a single DataFrame ({len(combined_df)} rows).")
        return list(
            pd.Series(combined_df['topic'].dropna().tolist())
            .drop_duplicates()
            .sort_values()
        )
    else:
        print("❌ No batches loaded.")
        return []

def _clean_json(txt: str) -> str:
    txt = txt.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-z]*\n|\n```$", "", txt, flags=re.I | re.S).strip()
    return txt

# --------------------------------------------------------------------
# helper: call model, validate JSON, retry on truncation/network
# --------------------------------------------------------------------
def _call_and_parse(
    model,
    messages: list[str],
    cfg,
    Schema,
    retries: int = 3,
    backoff: float = 2.0,
):
    for attempt in range(1, retries + 1):
        try:
            resp = model.generate_content(messages, generation_config=cfg)
            raw  = _clean_json(resp.text or "")
            return Schema.model_validate_json(raw)

        except ValidationError as ve:
            # likely ran out of output tokens
            if "EOF while parsing" in str(ve) and attempt < retries:
                # cfg.max_output_tokens = min(int(cfg.max_output_tokens * 1.5), 65_000)
                log.warning("JSON truncated (EOF). retrying (attempt %s/%s).",
                             attempt+1, retries)
            else:
                raise  # bubble up all other validation errors

        except Exception as e:
            if attempt == retries:
                raise
            log.warning("Model call failed (%s). Retrying in %.1fs (attempt %s/%s)...",
                        e, backoff, attempt+1, retries)
        time.sleep(backoff)
        
        
def process_reviews_in_batches(reviews_list, batch_size=200, output_dir='review_batches',model = None):
    # Ensure the output directory exists
    
    os.makedirs(output_dir, exist_ok=True)
    schema = ReviewsAnalyzerGem.model_json_schema()         

    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema,
    temperature=0, max_output_tokens=60000)
    topics_list = []
    df_list = []
    total_batches = (len(reviews_list) + batch_size - 1) // batch_size
    
    for i in range(0, len(reviews_list), batch_size):
        
        batch_number = i // batch_size + 1
        batch_filename = os.path.join(output_dir, f'review_batch_{batch_number}.csv')
        
        # Check if this batch has already been processed
        if os.path.exists(batch_filename):
            print(f"Batch {batch_number} already processed. Skipping...")
            continue
        
        batch = reviews_list[i:i + batch_size]
        if topics_list == []:
            if batch_number > 1:
                topics_list = load_batches_topics(output_dir, batch_number)
                gem_batch = [get_survey_second_pass_prompt(topics_list)] + [r["content"] for r in batch]
            else:
                gem_batch = [get_survey_first_pass_prompt()] + [r["content"] for r in batch]
        else:
            gem_batch = [get_survey_second_pass_prompt(topics_list)] + [r["content"] for r in batch]
        try:
            result = _call_and_parse(model, gem_batch, cfg, ReviewsAnalyzerGem)
        except Exception as fatal:
            log.error("Batch %s failed permanently: %s", batch_number, fatal)
            continue

        # Transform the result into a DataFrame
        data = []
        for comment in result.comments:
            for topic_detail in comment.topics:
                data.append({
                    'id': comment.id,
                    'sentiment': comment.sentiment,
                    'topic': topic_detail.topic,
                    'pain_points': ', '.join(topic_detail.pain_points) if topic_detail.pain_points else None,
                    'moments_of_delight': ', '.join(topic_detail.moments_of_delight) if topic_detail.moments_of_delight else None
                })

        df_batch = pd.DataFrame(data)
        topics_list = list(
            pd.Series(topics_list + df_batch['topic'].dropna().tolist())
            .drop_duplicates()
            .sort_values()
        )
        df_list.append(df_batch)
        
        # Save the current batch to a CSV file
        df_batch.to_csv(batch_filename, index=False)
        print(f"Batch {batch_number}/{total_batches} processed and saved.")

    # Concatenate all batch DataFrames into one if there were new batches processed
    final_df = pd.DataFrame()
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_csv(os.path.join(output_dir, 'all_reviews_combined.csv'), index=False)
        print("All batches combined and saved.")
    else:
        print("No new batches were processed.")
    
    return final_df

def process_reviews(reviews):
    # Filter out rows with no text or very short reviews
    
    reviews = reviews[['id', 'channel', 'comment']]
    reviews = reviews[reviews['comment'].notna()]  # Filter out no text reviews
    reviews = reviews[reviews['comment'].str.len() > 5]  # Filter out very short reviews
    

    
    # Ensure 'review_text' is of type string
    reviews['comment'] = reviews['comment'].astype(str)
    
    # Calculate number of words and tokens
    reviews['number_of_words'] = reviews['comment'].apply(lambda x: len(x.split()))
    reviews['number_of_tokens'] = reviews['comment'].apply(lambda x: len(word_tokenize(x)))
    
    return reviews

def prepare_reviews_for_openai(reviews):
    reviews_list = []
    for index, row in reviews.iterrows():
        review = f"Comment_id: {row['id']} \n channel:{row['channel']} \n Comment: {row['comment']}"
        reviews_list.append({"role": "user", "content": review})
    return reviews_list

def obtain_initial_sentiments(full_path,model):
    raw_reviews = pd.read_csv(full_path)
    reviews = process_reviews(raw_reviews)
    reviews_list = prepare_reviews_for_openai(reviews)
    openai_reviews = process_reviews_in_batches(reviews_list, batch_size=150, output_dir=f'output/2025-10-27_batches',model=model)
    openai_reviews.to_csv('./output/final_sentiment_analysis.csv', index=False,encoding='utf-8-sig')
    print("Final sentiment analysis completed and saved.")
    

if __name__ == "__main__":
    model = make_model(os.environ.get('GEMINI_API_KEY'))
    obtain_initial_sentiments("all_channels.csv",model)