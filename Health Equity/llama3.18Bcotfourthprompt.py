import torch, accelerate
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

def build_prompt(text):
    # System message
    system_message = "In this task, you will analyze a study abstract to determine if it qualifies as people-focused health equity scholarship."

    # Prompt with system message
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>
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
    - Final Answer: Provide either Yes or No, indicating whether the study qualifies as health equity scholarship."""


    # Add the user input
    prompt += f"Now, here is your input for the same task:\n**Input Text:**\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

if __name__ == "__main__":
    input_file = "/projects/ouzuner/fahmed34/Equity/validation_result/predicted_no_llama31_cotvalranprompt3cascade_001.csv"
    output_file = "/projects/ouzuner/fahmed34/Equity/validation_result/llama31_cotvalran3exprompt4cascaderun_001.csv"
    cache_dir = "/projects/ouzuner/fsakib/huggingface_models/model_files"

    now = datetime.now()
    print(f"Current DateTime = {now.strftime('%d/%m/%Y %H:%M:%S')}")
    eot_text = "<|eot_id|>"

    # Load the data
    df = pd.read_csv(input_file)
    notes = df['Abstract'].tolist()
    notes = df['Abstract'].astype(str).fillna("").tolist()
    # labels = df['PE_Label'].tolist()
    # note_ids = df['note_id'].tolist()

    # Initialize tokenizer and model
    model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print(f"Model Name: {model_path.split('/')[-1]}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        token=""
    )
    tokenizer.pad_token = tokenizer.eos_token

    accelerator = Accelerator()
    device = accelerator.device
    print(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        device_map="auto",
        offload_folder="offload",
        torch_dtype=torch.bfloat16,
        token=""
    )
    device = torch.device("cuda")

    # Prepare output DataFrame
    # df_results = pd.DataFrame({'note_id': note_ids, 'text': notes, 'PE_Label': labels})
    df_results = pd.DataFrame()
    df_results['Abstract'] = df['Abstract'].tolist()
    df_results['Health equity scholarship (yes or no)'] = df['Health equity scholarship (yes or no)'].tolist()

    # Define generation parameters
    generation_kwargs = {
        "min_length": 0,  # Corrected from -1
        "max_new_tokens": 4000,  # Adjusted to prevent memory issues
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }

    # Prepare expert texts
    max_length = tokenizer.model_max_length
    expert_texts = [build_prompt(note[:max_length]) for note in notes]
    responses = []

    # Process expert texts
    for expert_text in tqdm(expert_texts):
        try:
            # Encode the input text
            query_encoding = tokenizer(expert_text, return_tensors="pt", truncation=True).to(device)

            # Generate response
            response_tensor = model.generate(
                input_ids=query_encoding['input_ids'],
                attention_mask=query_encoding['attention_mask'],
                **generation_kwargs
            )
            
            # Decode response
            response = tokenizer.decode(response_tensor[0], skip_special_tokens=True)
            
            response = response.split(eot_text)[0]
            
            responses.append(response)

            # Debugging output
            print('-' * 100)
            print(f"Query:\n{expert_text}\n\n")
            print(f"Response:\n{response}")
            print('-' * 100)

        except Exception as e:
            print(f"Error processing text: {expert_text[:100]}... | Error: {e}")
            responses.append("Error generating response")  # Placeholder for failed responses

    # Add generated responses to the DataFrame
    df_results['Generated Response'] = responses

    # Save results
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    now = datetime.now()
    print(f"Finished DateTime = {now.strftime('%d/%m/%Y %H:%M:%S')}")
