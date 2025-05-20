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

    1. **Study Context or Setting:

    - Does the study involve underserved or resource-limited settings? Examples include:
        * Rural areas in the U.S. or the Global South.
        * Community hospitals, juvenile justice centers, or Title 1 schools.
        * Does the study focus on health conditions overwhelmingly concentrated in disenfranchised populations?
        * Age of participants alone does not determine health equity scholarship. Additionally, studies focused solely on physical health interventions do not qualify unless they address systemic inequities or disparities.

    **Final Evaluation:**

    - If the study meets any of the above criteria, it qualifies as health equity scholarship.
    - If ambiguity or partial focus exists, classify the study as "Yes" to avoid missing relevant work.
    - Final Answer: Provide either Yes or No, indicating whether the study qualifies as health equity scholarship.

    **Examples of Reasoning Process:** """

    examples = [
    {
        "abstract": "Introduction: Race and ethnicity are social constructs that are associated with meaningful health inequities. To address health disparities, it is essential to have valid, reliable race and ethnicity data. We compared child race and ethnicity as identified by the parent with that reported in the electronic health record (EHR). Methods: A convenience sample of parents of pediatric emergency department (PED) patients completed a tablet-based questionnaire (February-May 2021). Parents identified their child's race and ethnicity from options within a single category. We used chi-square to compare concordance between child race and ethnicity reported by the parent with that recorded in the EHR. Results: Of 219 approached parents, 206 (94%) completed questionnaires. Race and/or ethnicity were misidentified in the EHR for 56 children (27%). Misidentifications were most common among children whose parents identified them as multiracial (100% vs 15% of children identified as a single race, P < 0.001) or Hispanic (84% vs 17% of non-Hispanic children, P < 0.001), and children whose race and/or ethnicity differed from that of their parent (79% vs 18% of children with the same race and ethnicity as their parent, P < 0.001). Conclusion: In this PED, misidentification of race and ethnicity was common. This study provides the basis for a multifaceted quality improvement effort at our institution. The quality of child race and ethnicity data in the emergency setting warrants further consideration across health equity efforts.",
        "reasoning": [
            "1. **Study Context or Setting:** The study is conducted in a pediatric emergency department (PED), which serves as a critical access point for healthcare, particularly for marginalized populations. Emergency departments often provide care to individuals from underserved communities who may lack regular access to primary care.",
            "The study investigates race and ethnicity misclassification in electronic health records (EHRs), an issue that disproportionately affects children from disenfranchised groups, such as Hispanic and multiracial individuals. This aligns with health equity scholarship as it highlights a systemic issue in healthcare data accuracy, which can contribute to disparities in treatment and outcomes.",
            "While the study does not directly address an intervention or disparity in a low-resource setting, it examines a crucial structural factor—EHR misclassification—that affects healthcare quality in emergency settings, where disenfranchised populations are often overrepresented.",
            "**Final Evaluation:** The study qualifies as health equity scholarship because it focuses on data misclassification in emergency care settings, which can disproportionately impact healthcare quality for disenfranchised populations."
        ],
        "final_answer": "Yes"
    },
    {
        "abstract": "Herpes simplex virus type 2 (HSV-2) is common globally and contributes significantly to the risk of acquiring HIV-1, yet these two sexually transmitted infections have not been sufficiently characterized for sexual and gender minorities (SGM) across Sub-Saharan Africa. To help fill this gap, we performed a retrospective study using plasma and serum samples from 183 SGM enrolled at the Lagos site of the TRUST/RV368 cohort in Nigeria, assayed them for HSV-2 antibodies with the Kalon ELISA and plasma cytokines and chemokines with Luminex, and correlated the findings with HIV-1 viral loads (VLs) and CD4 counts. We found an overall HSV-2 prevalence of 36.6% (49.5% and 23.9% among SGM with and without HIV-1, respectively, p< .001). Moreover, HSV-2-positive status was associated with high circulating concentrations of CCL11 among antiretroviral therapy-treated (p= .031) and untreated (p= .015) participants, and with high concentrations of CCL2 in the untreated group (p= .004), independent of VL. Principal component analysis revealed a strong association of cytokines with HIV-1 VL independent of HSV-2 status. In conclusion, our study finds that HSV-2 prevalence among SGM with HIV-1 is twice as high than HSV-2 prevalence among SGM without HIV-1 in Lagos and suggests that this is associated with higher levels of certain systemic cytokines. Additional work is needed to further characterize the relationship between HSV-2 and HIV-1 in SGM and help develop targeted therapies for coinfected individuals.",
        "reasoning": [
            "1. **Study Context or Setting:** The study is conducted in Lagos, Nigeria, a region in Sub-Saharan Africa where healthcare resources may be limited, particularly for sexual and gender minorities (SGM), who already face systemic barriers to healthcare access.",
            "The study focuses on HIV-1 and HSV-2, two sexually transmitted infections that disproportionately affect marginalized populations, including SGM, in underserved settings. These conditions are recognized public health concerns in Sub-Saharan Africa, where access to testing, prevention, and treatment services is often inadequate.",
            "By examining co-infection rates and immune responses in an underserved population within a low-resource setting, the study highlights structural healthcare inequities and contributes to health equity scholarship by informing targeted interventions for vulnerable populations.",
            "**Final Evaluation:** The study qualifies as health equity scholarship because it investigates health disparities within an underserved setting, focusing on a disenfranchised population (SGM) disproportionately affected by HIV-1 and HSV-2."
        ],
        "final_answer": "Yes"
    },  
    {
        "abstract": "Background: Residential segregation is an important factor that negatively impacts cancer disparities, yet studies yield mixed results and complicate clear recommendations for policy change and public health intervention. In this study, we examined the relationship between local and Metropolitan Statistical Area (MSA) measures of Black isolation (segregation) and survival among older non-Hispanic (NH) Black women with breast cancer (BC) in the United States. We hypothesized that the influence of local isolation on mortality varies based on MSA isolation-specifically, that high local isolation may be protective in the context of highly segregated MSAs, as ethnic density may offer opportunities for social support and buffer racialized groups from the harmful influences of racism. Methods: Local and MSA measures of isolation were linked by Census Tract (CT) with a SEER-Medicare cohort of 5,231 NH Black women aged 66-90 years with an initial diagnosis of stage I-IV BC in 2007-2013 with follow-up through 2018. Proportional and cause-specific hazards models and estimated marginal means were used to examine the relationship between local and MSA isolation and all-cause and BC-specific mortality, accounting for covariates (age, comorbidities, tumor stage, and hormone receptor status). Findings: Of 2,599 NH Black women who died, 40.0% died from BC. Women experienced increased risk for all-cause mortality when living in either high local (HR = 1.20; CI = 1.08-1.33; p < 0.001) or high MSA isolation (HR = 1.40; CI = 1.17-1.67; p < 0.001). A similar trend existed for BC-specific mortality. Pairwise comparisons for all-cause mortality models showed that high local isolation was hazardous in less isolated MSAs but was not significant in more isolated MSAs. Interpretation: Both local and MSA isolation are independently associated with poorer overall and BC-specific survival for older NH Black women. However, the impact of local isolation on survival appears to depend on the metropolitan area's level of segregation. Specifically, in highly segregated MSAs, living in an area with high local isolation is not significantly associated with poorer survival. While the reasons for this are not ascertained in this study, it is possible that the protective qualities of ethnic density (e.g., social support and buffering from experiences of racism) may have a greater role in more segregated MSAs, serving as a counterpart to the hazardous qualities of local isolation. More research is needed to fully understand these complex relationships. Funding: National Cancer Institute.",
        "reasoning": [
            "1. **Study Context or Setting:** The study focuses on the impact of residential segregation on breast cancer survival among non-Hispanic (NH) Black women, a population that faces systemic healthcare disparities in the United States.",
            "Residential segregation is a key structural determinant of health that often leads to reduced access to high-quality healthcare, economic opportunities, and social resources. The study examines how different levels of segregation at local and metropolitan levels affect cancer mortality, highlighting systemic inequities.",
            "By linking Census Tract data with SEER-Medicare records, the study analyzes structural barriers affecting health outcomes, which aligns with health equity scholarship. The findings contribute to understanding how social and environmental factors shape disparities in breast cancer survival.",
            "**Final Evaluation:** The study qualifies as health equity scholarship because it investigates how residential segregation, a systemic factor, influences health disparities in cancer survival among a historically marginalized population."
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
    input_file = "/projects/ouzuner/fahmed34/Equity/validation_result/predicted_no_llama31_cotvalranprompt1cascade_001.csv"
    output_file = "/projects/ouzuner/fahmed34/Equity/validation_result/llama31_cotvalran3exprompt2cascaderun_001.csv"
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
