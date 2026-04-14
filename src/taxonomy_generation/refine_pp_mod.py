import nltk
import os
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from pydantic import BaseModel, Field
from typing import List, Optional
from typing import List, Literal
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from dotenv import load_dotenv
load_dotenv()
from google.genai import types
from pydantic import ValidationError
import re,logging,time
from pydantic import BaseModel, Field
from typing import List, Optional
from typing import List, Literal
log = logging.getLogger(__name__)
from tqdm import tqdm
import pandas as pd
from src.core.llm import make_model

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
        

class SubtopicDetails(BaseModel):
    type: str
    gaca_service: str
    cleaned_subtopic: str
    old_subtopic : str

class Subtopics(BaseModel):
    Subtopics: List[SubtopicDetails]
    
sys_prompt="""
You are a text-cleaning model used in GACA’s Service Quality pipeline.
Your job is to take messy subtopic strings and convert them into a clean, human-readable English subtopic by combining the two parts of the phrase into a simple, meaningful title-case expression.

**Transformation Rules**
    1- Output only the final combined phrase.
        - No explanation
        - No punctuation
        - No examples
        - No notes

    2- Split the subtopic by hyphens (“-”) or “–”
        - Trim whitespace
        - Only keep the two meaningful components.

    3-Reorder intelligently if needed
        - If the first part is descriptive and the second part is the subject, output: <Descriptor> <Subject>
            - Example: “Damaged - Cargo” → “Damaged Cargo”
            - Example: “Late - Boarding” → “Late Boarding”

        - If the first part is a phrase and it needs reordering, keep the adjective and object but reorder to form a meaningful title. examples:
            - “Great Experience with - Boarding” → “Great Boarding Experience”
            - “Rude Agents - Agent Interaction” → “Rude Agent Interaction”
        
        - If the combined phrase is awkward, rephrase to be more natural while keeping the original meaning.
            - for example, “Poor Cleanliness - Lounge” becomes “Poor Lounge Cleanliness"
            - "dirty - Lounge Access" becomes "Dirty Lounge Facilities"
            - "poorly placed - Screen Location" becomes "Poor Screen Placement"
        - Always make sure that the final phrase is fluent and natural in English. Even if you would change the wording slightly, keep the original intent.
            
        
    4-Capitalize Each Word (Title Case)
    
    5-The output must be direct, and meaningful, respecting the category (Airline/Airport) and GACA service context.
    
Example 1 Input:

type: "Airline"
Gaca_Service: "Air freight services"
subtopic: “Great Experience with - Boarding”

Example 1 Output:
{
    "type": "Airline",
    "gaca_service": "Air freight services",
    "old_subtopic":"Great Experience with - Boarding",
    "cleaned_subtopic":"Great Boarding Experience"
}

"""
def prepare_items(df):
    inputs_list = []
    df['combined'] = df['Mod'] + ' - ' + df['root_subtopic']
    for index, row in df.iterrows():
        review = f"type: {row['type']}\nGaca_Service: {row['topic']}\nsubtopic: {row['combined']}"
        inputs_list.append({"role": "user", "content": review})
    return inputs_list

def prepare_missing_items(df):
    inputs_list = []
    for index, row in df.iterrows():
        review = f"type: {row['type']}\nGaca_Service: {row['topic']}\nsubtopic: {row['subtopic']}"
        inputs_list.append({"role": "user", "content": review})
    return inputs_list


def process_subtopics( model, batch_size,inputs,output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    schema = Subtopics.model_json_schema()         

    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema,
    temperature=0, max_output_tokens=60000)
    
    df_list = []
    
    total_batches = (len(inputs) + batch_size - 1) // batch_size
    
    for i in range(0, len(inputs), batch_size):
        batch_number = i // batch_size + 1
        batch_filename = os.path.join(output_folder, f'review_batch_{batch_number}.csv')
        
        # Check if this batch has already been processed
        if os.path.exists(batch_filename):
            print(f"Batch {batch_number} already processed. Skipping...")
            continue
        
        batch = inputs[i:i + batch_size]
        
        gem_batch = [sys_prompt] + [r["content"] for r in batch]

        try:
            result = _call_and_parse(model, gem_batch, cfg, Subtopics)
        except Exception as fatal:
            log.error("Batch %s failed permanently: %s", fatal)
            continue

        # Transform the result into a DataFrame
        data = []
        for subtopic_detail in result.Subtopics:
            data.append({
                'type': subtopic_detail.type,
                'gaca_service': subtopic_detail.gaca_service,
                'old_subtopic': subtopic_detail.old_subtopic, 
                'cleaned_subtopic': subtopic_detail.cleaned_subtopic,
            })

        df_batch = pd.DataFrame(data)
        df_list.append(df_batch)
        
        # Save the current batch to a CSV file
        df_batch.to_csv(batch_filename, index=False)
        print(f"Batch {batch_number}/{total_batches} processed and saved.")

    # Concatenate all batch DataFrames into one if there were new batches processed
    final_df = pd.DataFrame()
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_csv('./refined_subtopics.csv', index=False)
        print("All batches combined and saved.")
    else:
        print("No new batches were processed.")
    
    return final_df

if __name__ == "__main__":
    model = make_model(os.environ.get('GEMINI_API_KEY'))
    df = pd.read_csv('./missing_subtopics.csv')
    inputs = prepare_missing_items(df)
    process_subtopics(model, 125,inputs,'./missing_refined_subtopic_batches2')