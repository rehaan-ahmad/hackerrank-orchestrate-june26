import pandas as pd
import os
import google.generativeai as genai
from typing import List, Dict, Any
import json
from dotenv import load_dotenv
import logging
import time
from random import uniform

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def configure_genai():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')

def load_dataset():
    try:
        claims_df = pd.read_csv('dataset/claims.csv')
        history_df = pd.read_csv('dataset/user_history.csv')
        reqs_df = pd.read_csv('dataset/evidence_requirements.csv')
        return claims_df, history_df, reqs_df
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        raise

def get_context(user_id, claim_object, history_df, reqs_df):
    # User History
    user_row = history_df[history_df['user_id'] == user_id]
    history = user_row.iloc[0].to_dict() if not user_row.empty else {}
    
    # Evidence Requirements
    relevant_reqs = reqs_df[(reqs_df['claim_object'] == claim_object) | (reqs_df['claim_object'] == 'all')]
    reqs = relevant_reqs.to_dict('records')
    
    return history, reqs

def process_claim(model, row, history_df, reqs_df):
    user_id = row['user_id']
    image_paths = row['image_paths'].split(';')
    user_claim = row['user_claim']
    claim_object = row['claim_object']
    
    history, reqs = get_context(user_id, claim_object, history_df, reqs_df)
    
    images = []
    for path in image_paths:
        full_path = os.path.join('dataset', path)
        if os.path.exists(full_path):
            try:
                with open(full_path, "rb") as f:
                    img_data = f.read()
                images.append({'mime_type': 'image/jpeg', 'data': img_data})
            except Exception as e:
                logger.warning(f"Could not read image {full_path}: {e}")

    prompt = f"""
    You are an expert insurance claims adjuster. Analyze the following damage claim.
    
    OBJECT: {claim_object}
    USER CLAIM: {user_claim}
    USER HISTORY: {history}
    MINIMUM EVIDENCE REQUIREMENTS: {reqs}
    
    TASK:
    1. Verify if the provided images meet the evidence requirements.
    2. Identify the specific issue type and the affected part of the {claim_object}.
    3. Determine if the claim is supported by visual evidence, contradicted, or if there is not enough information.
    4. Identify any risk flags (blurry, wrong angle, history risks, etc.).
    5. Estimate severity.
    
    CONSTRAINTS:
    - issue_type must be one of: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown.
    - claim_status must be one of: supported, contradicted, not_enough_information.
    - severity must be one of: none, low, medium, high, unknown.
    - risk_flags must be semicolon-separated from the allowed list, or 'none'.
    - supporting_image_ids should be the filename without extension (e.g., 'img_1').
    
    OUTPUT FORMAT:
    Return ONLY a valid JSON object with these keys:
    {{
        "evidence_standard_met": boolean,
        "evidence_standard_met_reason": "string",
        "risk_flags": "string",
        "issue_type": "string",
        "object_part": "string",
        "claim_status": "string",
        "claim_status_justification": "string",
        "supporting_image_ids": "string",
        "valid_image": boolean,
        "severity": "string"
    }}
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            content = [prompt] + images
            response = model.generate_content(content)
            
            text = response.text.strip()
            if text.startswith('```json'):
                text = text[7:-3].strip()
            elif text.startswith('```'):
                text = text[3:-3].strip()
            
            res_json = json.loads(text)
            return {**row, **res_json}
        except Exception as e:
            if "429" in str(e):
                wait_time = (2 ** attempt) + uniform(0, 1)
                logger.warning(f"Quota exceeded. Retrying in {wait_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                logger.error(f"Error processing claim {user_id}: {e}")
                break
                
    return {
        **row,
        'evidence_standard_met': False,
        'evidence_standard_met_reason': "Max retries exceeded or processing error",
        'risk_flags': 'none',
        'issue_type': 'unknown',
        'object_part': 'unknown',
        'claim_status': 'not_enough_information',
        'claim_status_justification': 'Processing failed after retries',
        'supporting_image_ids': 'none',
        'valid_image': False,
        'severity': 'unknown'
    }

def main():
    logger.info("Starting claim processing pipeline...")
    try:
        model = configure_genai()
        claims_df, history_df, reqs_df = load_dataset()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        return

    results = []
    total = len(claims_df)
    for i, row in claims_df.iterrows():
        logger.info(f"Processing claim {i+1}/{total} (User: {row['user_id']})")
        results.append(process_claim(model, row, history_df, reqs_df))
        # Respect basic rate limits for free tier
        time.sleep(2) 
    
    output_df = pd.DataFrame(results)
    cols = [
        'user_id', 'image_paths', 'user_claim', 'claim_object', 'evidence_standard_met', 
        'evidence_standard_met_reason', 'risk_flags', 'issue_type', 'object_part', 
        'claim_status', 'claim_status_justification', 'supporting_image_ids', 'valid_image', 'severity'
    ]
    for col in cols:
        if col not in output_df.columns:
            output_df[col] = 'unknown' if col != 'evidence_standard_met' else False

    output_df[cols].to_csv('output.csv', index=False)
    logger.info("Successfully generated output.csv")

if __name__ == "__main__":
    main()
