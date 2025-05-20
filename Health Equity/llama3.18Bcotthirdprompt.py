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

    1. **Examination of Discrimination or Oppression:

    - Does the study explicitly explore the impact of racism, sexism, ableism, xenophobia, or other forms of discrimination or oppression?
        * Does it examine strategies to reduce bias, harm, or inequities (e.g., training programs for providers to improve interactions with marginalized populations)?
        * Age of participants alone does not determine health equity scholarship. Additionally, studies focused solely on physical health interventions do not qualify unless they address systemic inequities or disparities.
    
    **Final Evaluation:**

    - If the study meets any of the above criteria, it qualifies as health equity scholarship.
    - If ambiguity or partial focus exists, classify the study as "Yes" to avoid missing relevant work.
    - Final Answer: Provide either Yes or No, indicating whether the study qualifies as health equity scholarship.

    **Examples of Reasoning Process:** """

    examples = [
    {
        "abstract": "Previous research suggests that routine psychosocial care for adolescents with attention-deficit/hyperactivity disorder (ADHD) is an eclectic and individualized mix of diluted evidence-based practices (EBPs) and low-value approaches. This study evaluated the extent to which a community-delivered EBP and usual care (UC) for adolescents with ADHD produce differential changes in theorized behavioral, psychological, and cognitive mechanisms of ADHD. A randomized community-based trial was conducted with double randomization of adolescent and community therapists to EBP delivery supports (Supporting Teens' Autonomy Daily [STAND]) versus UC delivery. Participants were 278 culturally diverse adolescents (ages 11-17) with ADHD and caregivers. Mechanistic outcomes were measured at baseline, post-treatment, and follow-up using parent-rated, observational, and task-based measures. Results using linear mixed models indicated that UC demonstrated superior effects on parent-rated and task-based executive functioning relative to STAND. However, STAND demonstrated superior effects on adolescent motivation and reducing parental intrusiveness relative to UC when it was delivered by licensed therapists. Mechanisms of community-delivered STAND and UC appear to differ. UC potency may occur through improved executive functioning, whereas STAND potency may occur through improved teen motivation and reducing low-value parenting practices. However, when delivered by unlicensed, community-based therapists, STAND did not enact proposed mechanisms. Future adaptations of community-delivered EBPs for ADHD should increase supports for unlicensed therapists, who comprise the majority of the community mental health workforce.",
        "reasoning": [
            "1. **Examination of Discrimination or Oppression:** The study highlights disparities in community mental health treatment by examining differences in therapy effectiveness when delivered by licensed vs. unlicensed therapists. The findings suggest that unlicensed community-based therapists, who make up the majority of the workforce, may not have adequate training or support, leading to lower effectiveness of evidence-based interventions like STAND.",
            "The study also touches on **systemic inequities** in access to quality ADHD care, as **culturally diverse adolescents** receiving care from unlicensed therapists may experience less effective treatment. This reflects broader disparities in the mental healthcare system, particularly for families who rely on community-based services.",
            "Additionally, the study suggests a **need for structural improvements**, including better support for unlicensed therapists, which aligns with efforts to reduce harm and inequities in mental healthcare. By identifying barriers that disproportionately impact underserved groups, the study contributes to discussions on reducing disparities in ADHD treatment.",
            "**Final Evaluation:** The study qualifies as health equity scholarship because it examines systemic barriers in community mental health, highlights inequities in therapist training and effectiveness, and suggests structural improvements to reduce disparities in ADHD care."
        ],
        "final_answer": "Yes"
    },

    {
        "abstract": "Objective: Children with low income and minority race and ethnicity have worse hospital outcomes due partly to systemic and interpersonal racism causing communication and system barriers. We tested the feasibility and acceptability of a novel inpatient communication-focused navigation program. Methods: Multilingual design workshops with parents, providers, and staff created the Family Bridge Program. Delivered by a trained navigator, it included 1) hospital orientation; 2) social needs screening and response; 3) communication preference assessment; 4) communication coaching; 5) emotional support; and 6) a post-discharge phone call. We enrolled families of hospitalized children with public or no insurance, minority race or ethnicity, and preferred language of English, Spanish, or Somali in a single-arm trial. We surveyed parents at enrollment and 2 to 4 weeks post-discharge, and providers 2 to 3 days post-discharge. Survey measures were analyzed with paired t tests. Results: Of 60 families enrolled, 57 (95%) completed the follow-up survey. Most parents were born outside the United States (60%) with a high school degree or less (60%). Also, 63% preferred English, 33% Spanish, and 3% Somali. The program was feasible: families received an average of 5.3 of 6 components; all received >2. Most caregivers (92%) and providers (81% [30/37]) were 'very satisfied.' Parent-reported system navigation improved from enrollment to follow-up (+8.2 [95% confidence interval 2.9, 13.6], P = .003; scale 0-100). Spanish-speaking parents reported decreased skills-related barriers (-18.4 [95% confidence interval -1.8, -34.9], P = .03; scale 0-100). Conclusions: The Family Bridge Program was feasible, acceptable, and may have potential for overcoming barriers for hospitalized children at risk for disparities.",
        "reasoning": [
            "1. **Examination of Discrimination or Oppression:** The study directly examines systemic and interpersonal racism as a contributing factor to worse hospital outcomes for children from low-income and minority racial and ethnic backgrounds. It acknowledges that structural discrimination leads to communication barriers and disparities in hospital care.",
            "The Family Bridge Program is designed to address these inequities by improving communication, social support, and system navigation for families with limited English proficiency and lower income. By focusing on marginalized communities, including Spanish and Somali speakers, the study highlights how linguistic and economic barriers intersect with systemic racism in healthcare settings.",
            "The study also evaluates whether a trained navigator can mitigate these disparities through interventions such as hospital orientation, social needs screening, and post-discharge follow-up. The significant improvement in navigation and reduction in skill-related barriers among Spanish-speaking parents suggests that targeted strategies can reduce inequities in hospital care.",
            "**Final Evaluation:** The study qualifies as health equity scholarship under Prompt 3 because it explicitly examines the impact of systemic racism in hospital care and evaluates a structured intervention aimed at reducing bias, harm, and disparities in healthcare access and outcomes."
        ],
        "final_answer": "Yes"
    }
  
    
]

    # Add examples to the prompt
    for idx, example in enumerate(examples, 1):
        prompt += f"**Example {idx}:**\n" \
                  f"**Input Text:**\n\"{example['abstract']}\"\n\n" \
                  f"**Step-by-Step Reasoning:**\n"
        for reasoning_step in example["reasoning"]:
            prompt += f"{reasoning_step}\n"
        prompt += f"\n**Final Answer:** {example['final_answer']}\n\n"

    # Add the user input
    prompt += f"Now, here is your input for the same task:\n**Input Text:**\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

if __name__ == "__main__":
    input_file = "/projects/ouzuner/fahmed34/Equity/validation_result/predicted_no_llama31_cotvalranprompt2cascade_001.csv"
    output_file = "/projects/ouzuner/fahmed34/Equity/validation_result/llama31_cotvalran3exprompt3cascaderun_001.csv"
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
