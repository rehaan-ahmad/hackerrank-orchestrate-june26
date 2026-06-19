import pandas as pd
import os
import google.generativeai as genai
from typing import List, Dict
import json
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure Gemini API
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

def load_csv(path):
    return pd.read_csv(path)

def get_user_history(user_id, history_df):
    user_row = history_df[history_df['user_id'] == user_id]
    if not user_row.empty:
        return user_row.iloc[0].to_dict()
    return {}

def get_evidence_reqs(object_type, reqs_df):
    relevant = reqs_df[(reqs_df['claim_object'] == object_type) | (reqs_df['claim_object'] == 'all')]
    return relevant.to_dict('records')

def process_claim(row, history_df, reqs_df):
    user_id = row['user_id']
    image_paths = row['image_paths'].split(';')
    user_claim = row['user_claim']
    claim_object = row['claim_object']
    
    history = get_user_history(user_id, history_df)
    reqs = get_evidence_reqs(claim_object, reqs_df)
    
    images = []
    for path in image_paths:
        full_path = os.path.join('dataset', path)
        if os.path.exists(full_path):
            img = genai.upload_file(path=full_path)
            images.append(img)

    prompt = f"""
    Analyze a damage claim for a {claim_object}.
    
    User Claim: {user_claim}
    User History: {history}
    Minimum Evidence Requirements: {reqs}
    
    Evaluate the images and provide the output in JSON format with these keys:
    - evidence_standard_met (boolean)
    - evidence_standard_met_reason (string)
    - risk_flags (semicolon-separated string or 'none')
    - issue_type (one of: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown)
    - object_part (valid part for {claim_object} or 'unknown')
    - claim_status (one of: supported, contradicted, not_enough_information)
    - claim_status_justification (string)
    - supporting_image_ids (semicolon-separated image IDs or 'none')
    - valid_image (boolean)
    - severity (one of: none, low, medium, high, unknown)
    
    Return ONLY JSON.
    """
    
    try:
        response = model.generate_content([prompt, *images])
        # Clean JSON response from Gemini (remove markdown code blocks)
        text = response.text.strip()
        if text.startswith('```json'):
            text = text[7:-3].strip()
        elif text.startswith('```'):
            text = text[3:-3].strip()
        
        res_json = json.loads(text)
        return {**row, **res_json}
    except Exception as e:
        print(f"Error processing claim {user_id}: {e}")
        return {
            **row,
            'evidence_standard_met': False,
            'evidence_standard_met_reason': str(e),
            'risk_flags': 'none',
            'issue_type': 'unknown',
            'object_part': 'unknown',
            'claim_status': 'not_enough_information',
            'claim_status_justification': 'Error in processing',
            'supporting_image_ids': 'none',
            'valid_image': False,
            'severity': 'unknown'
        }

def main():
    claims_df = load_csv('dataset/claims.csv')
    history_df = load_csv('dataset/user_history.csv')
    reqs_df = load_csv('dataset/evidence_requirements.csv')
    
    results = []
    for _, row in claims_df.iterrows():
        results.append(process_claim(row, history_df, reqs_df))
    
    output_df = pd.DataFrame(results)
    
    # Ensure columns are in the required order
    cols = ['user_id', 'image_paths', 'user_claim', 'claim_object', 'evidence_standard_met', 
            'evidence_standard_met_reason', 'risk_flags', 'issue_type', 'object_part', 
            'claim_status', 'claim_status_justification', 'supporting_image_ids', 'valid_image', 'severity']
    
    output_df[cols].to_csv('output.csv', index=False)
    print("Successfully generated output.csv")

if __name__ == "__main__":
    main()
