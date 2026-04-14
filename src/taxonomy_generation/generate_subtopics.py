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


class SubtopicsResponse(BaseModel):
    root_subtopic: str
    positive_mods: List[str] = Field(default_factory=list)
    negative_mods: List[str] = Field(default_factory=list)
    
class SubtopicDetails(BaseModel):
    type: str
    gaca_services: str
    subtopics: List[SubtopicsResponse] = Field(default_factory=list)

system_prompt = """
    You are a domain-agnostic topic inducer helping categorize customer-experience feedback for Saudi Arabia’s civil-aviation service context (GACA) that must output a single, valid JSON object only.
    Given a single input object with fields : 
        - type: "Airline" or "Airport"
        - Gaca_Service: e.g. "Cargo", "Fligt", "Boarding" .....
    produce a generic,compact and reusable taxonomy below that service:
        - A list of 2–5 root_subtopics (generic and reusable depending on the type provided).
        - For each root_subtopic, two short lists of 2-5:
            - positive_mods: positive adjectives/short experience phrases customers might express.
            - negative_mods: negative adjectives/short experience phrases customers might express.
        These “mods” are short, sentiment-bearing descriptors (1–3 words, max 4) like “on time”, “helpful staff”, “delayed”, “lost item”. Avoid full sentences.    
        The "mods" should be used as add-ons to the root_subtopic to form meaningful phrases (e.g., "On time - Cargo", "Delayed - Boarding").
    
    Generation Rules:
        - The "mods" should be used as add-ons to the root_subtopic to form meaningful phrases (e.g., "On time - Cargo", "Delayed - Boarding").
        - If a mod is talking about experience with a service, it should be in the format "Great Experience with" or "Poor Experience with", not detailed experiences as "smooth", "efficient","quick","hassle-free" and so on.        - Generic/reusable: don’t reference a specific airport, brand, or country.
        - Keep phrases short; Title Case not required; no punctuation except hyphens or slashes if needed.
        - No duplicates across mods; avoid near-duplicates (e.g., “on time” vs “on-time”) and avoid semantic duplicates.
        - Subtopics should reflect process steps, touchpoints, service qualities, logistics, support, info, pricing/billing, policies, access, facilities, safety/compliance, etc.—as relevant to the service.
        - Mods should be descriptors, not causes; e.g., “clear instructions” is good, “because the agent…” is not.
    
    Return only JSON (no explanations), using double quotes and ASCII.
    
    Schema:
    {
    "type": "<echo input.type>",
    "gaca_services": "<echo input.Gaca_Service>",
    "subtopics": [
        {
        "root_subtopic": "<short noun phrase>",
        "positive_mods": ["<short phrase>", "..."],
        "negative_mods": ["<short phrase>", "..."]
        }]}

    Example Input:
    
    type: "Airline"
    Gaca_Service: "Air freight services"
  
    Example Output:
    
    {'type': 'Airline', 'gaca_services':'Air freight services',
    'subtopics':[
        {'root_subtopic':'Cargo','positive_mods':["On time","Great Experience with"],'negative_mods':['Damaged','Delayed','Lost','Poor Experience with']},
        {'root_subtopic':'Flight','positive_mods':['On time'],'negative_mods':['Cancelled','Delayed','Changed']},
        {'root_subtopic':'Boarding','positive_mods':['Great Experience with'],'negative_mods':['Delayed','Denied','Poor Experience with']}
        ]       
       """

def process_topics( model, topic_types):
    # Ensure the output directory exists
    
    
    schema = SubtopicDetails.model_json_schema()         

    cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema,
    temperature=0, max_output_tokens=60000)
    
    df_list = []
    
    for row in tqdm(topic_types.itertuples(), total=topic_types.shape[0], desc="Processing pairs"):
        
        
        item = f"type: {row.TYPE} \n Gaca_Service:{row.CATEGORY_NAME}"
        
        gem_batch = [system_prompt] + [item]
        try:
            result = _call_and_parse(model, gem_batch, cfg, SubtopicDetails)
        except Exception as fatal:
            log.error("Batch %s failed permanently: %s", fatal)
            continue

        # Transform the result into a DataFrame
        data = []
        for subtopic_detail in result.subtopics:
            data.append({
                'type': result.type,
                'topic': result.gaca_services,
                'root_subtopic': subtopic_detail.root_subtopic, 
                'negative_mods': ', '.join(subtopic_detail.negative_mods) if subtopic_detail.negative_mods else None,
                'positive_mods': ', '.join(subtopic_detail.positive_mods) if subtopic_detail.positive_mods else None
            })

        df_batch = pd.DataFrame(data)
        df_list.append(df_batch)
        
        # Save the current batch to a CSV file
        # df_batch.to_csv(batch_filename, index=False)
        # print(f"Batch {batch_number}/{total_batches} processed and saved.")

    # Concatenate all batch DataFrames into one if there were new batches processed
    final_df = pd.DataFrame()
    if df_list:
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_csv('./topic_subtopic_refined.csv', index=False)
        print("All batches combined and saved.")
    else:
        print("No new batches were processed.")
    
    return final_df

if __name__ == "__main__":
    model = make_model(os.environ.get('GEMINI_API_KEY'))
    df = pd.read_csv('../actual_data/website_categories.csv')
    topics = df[['CATEGORY_NAME','TYPE']].dropna().drop_duplicates().reset_index(drop=True)
    print("Processing total topics:", topics.shape[0])
    process_topics(model,topics)