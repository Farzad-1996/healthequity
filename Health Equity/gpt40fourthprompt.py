import openai
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import re
import string
import matplotlib.pyplot as plt
import random

# Set up OpenAI API key
openai.api_key = ''


file_path = '/projects/ouzuner/fahmed34/Equity/results/predicted_no_gpt4othirdprompt_001.csv'


df = pd.read_csv(file_path)
data= df

# Define the prompt template
prompt_template = """You are an expert in analyzing study abstracts to determine if they qualify as people-focused health equity scholarship.

Follow these structured steps for reasoning:

    1. **Research Methods or Equity Focus:

    - Does the study use methods or strategies to increase equitable engagement or access among disenfranchised groups? Examples include:
        * Community-based participatory research.
        * Inclusion of culturally or linguistically adapted tools (e.g., non-English surveys).
        * Descriptions of experiences in underserved or resource-limited settings.
        * Age of participants alone does not determine health equity scholarship. Additionally, studies focused solely on physical health interventions do not qualify unless they address systemic inequities or disparities.
    **Final Evaluation:**

    - If the study meets any of the above criteria, it qualifies as health equity scholarship.
    - If ambiguity or partial focus exists, classify the study as "Yes" to avoid missing relevant work.
    - Final Answer: Provide either Yes or No, indicating whether the study qualifies as health equity scholarship.

    ### **Instructions for Output:**
    1. **Your response must start with:** `Final Answer: Yes` or `Final Answer: No`
    2. **Then, provide the reasoning behind the classification.**
    3. **Do not include any other text, explanations, or disclaimers outside of the Final Answer and reasoning.**
    
---

Here is the abstract for classification:

**Abstract:**  
"{}"
"""



def zero_shot_classify_batch(texts):
    prompts = [prompt_template.format(text) for text in texts]
    predictions = []
    
    for prompt in prompts:
        retries = 3  
        delay = 10   
        
        for attempt in range(retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=30,
                    temperature=0.0
                )
                prediction = response['choices'][0]['message']['content'].strip()
                predictions.append(prediction)
                
                delay = 10  # Reset delay after a successful call
                break  
            
            except openai.RateLimitError:
                print(f"⚠️ Rate limit reached. Retrying in {delay} seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(delay + random.uniform(1, 3))  # Add slight randomness to avoid collisions
                delay *= 2  
            
            except openai.OpenAIError as e:
                print(f"🚨 OpenAI API Error: {e}")
                predictions.append("Error")
                break  
            
            except Exception as e:
                print(f"🔥 Unexpected error: {e}")
                predictions.append("Error")
                break  
        
        else:  
            print("❌ Max retries reached. Moving to the next prompt.")
            predictions.append("Error")

    return predictions

# Process data in smaller batches
batch_size = 10  # Reduced batch size for reliability
predictions = []

for i in range(0, len(data), batch_size):
    batch_texts = data['Abstract'][i:i+batch_size].tolist()
    batch_predictions = zero_shot_classify_batch(batch_texts)
    predictions.extend(batch_predictions)
    time.sleep(1)  # Brief pause to manage rate limits

# Ensure predictions length matches data length
if len(predictions) < len(data):
    predictions.extend(["Error"] * (len(data) - len(predictions)))

# Add predictions to data and save to CSV
data['prediction'] = predictions
data.to_csv("/projects/ouzuner/fahmed34/Equity/results/classified_data_gpt4o_fourthpromt-001.csv", index=False)

print("Predictions saved to classified_data.csv")
