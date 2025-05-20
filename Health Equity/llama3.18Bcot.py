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

    1. **Focus on Disenfranchised Individuals/Populations:**
    - Does the study intentionally include or focus on disenfranchised populations? Disenfranchised populations include:
        * BIPOC (Black, Indigenous, People of Color), including Hispanic/Latino/a/x individuals.
        * LGBTQ+, gender diverse, or transgender individuals.
        * Individuals with disabilities or intellectual/developmental delays.
        * Socioeconomically disadvantaged groups (e.g., poverty, food insecurity, lack of healthcare access).
        * Individuals who speak a language other than English.
        * Populations receiving care in underserved settings (e.g., rural areas, Federally Qualified Health Centers, community hospitals, Title 1 schools, or low-and-middle-income countries).
        * Groups disproportionately affected by specific conditions (e.g., sickle cell disease, malaria, HIV).
        * Women (to consider sex differences) are also regarded as disenfranchised individuals in this context.
    - Does the study explore health disparities, inequities, or differences in health outcomes based on these identities?

    2. **Study Context or Setting:**
    - Does the study involve underserved or resource-limited settings? Examples include:
        * Rural areas in the U.S. or the Global South.
        * Community hospitals, juvenile justice centers, or Title 1 schools.
    - Does the study focus on health conditions overwhelmingly concentrated in disenfranchised populations?

    3. **Examination of Discrimination or Oppression:**
    - Does the study explicitly explore the impact of racism, sexism, ableism, xenophobia, or other forms of discrimination or oppression?
    - Does it examine strategies to reduce bias, harm, or inequities (e.g., training programs for providers to improve interactions with marginalized populations)?

    4. **Research Methods or Equity Focus:**
    - Does the study use methods or strategies to increase equitable engagement or access among disenfranchised groups? Examples include:
        * Community-based participatory research.
        * Inclusion of culturally or linguistically adapted tools (e.g., non-English surveys).
        * Descriptions of experiences in underserved or resource-limited settings.

    5. **Additional Considerations:**
    - Age of participants is not a criterion for determining health equity scholarship.
    - Physical health interventions targeting specific health conditions are not considered part of health equity unless they address systemic inequities or disparities.

    6. **Final Evaluation:**
    - If the study meets any of the above criteria, it qualifies as health equity scholarship.
    - If ambiguity or partial focus exists, classify the study as 'Yes' to avoid missing relevant work.

    **Examples of Reasoning Process:** """

    examples = [
    {
        "abstract": "Predicting long-term kidney allograft failure is an unmet need for clinical care and clinical trial optimization in children. We aimed to validate a kidney allograft failure risk prediction system in a large international cohort of pediatric kidney transplant recipients. Patients from 20 centers in Europe and the United States, transplanted between 2004 and 2017, were included. Allograft assessment included estimated glomerular filtration rate, urine protein-to-creatinine ratio, circulating antihuman leukocyte antigen donor-specific antibody, and kidney allograft histology. Individual predictions of allograft failure were calculated using the integrative box (iBox) system. Prediction performances were assessed using discrimination and calibration. The allograft evaluations were performed in 706 kidney transplant recipients at a median time of 9.1 (interquartile range, 3.3-19.2) months posttransplant; mean estimated glomerular filtration rate was 68.7 ± 28.1 mL/min/1.73 m2, and median urine protein-to-creatinine ratio was 0.1 (0.0-0.4) g/g, and 134 (19.0%) patients had antihuman leukocyte antigen donor-specific antibodies. The iBox exhibited accurate calibration and discrimination for predicting the outcomes up to 10 years after evaluation, with a C-index of 0.81 (95% confidence interval, 0.75-0.87). This study confirms the generalizability of the iBox to predict long-term kidney allograft failure in children, with performances similar to those reported in adults. These results support the use of the iBox to improve patient monitoring and facilitate clinical trials in children.",
        "reasoning": [
            "1. **Focus on Disenfranchised Individuals/Populations:** The study focuses on pediatric kidney transplant recipients, who may be considered medically vulnerable. However, it does not explicitly target a systematically disenfranchised group such as BIPOC, LGBTQ+, or socioeconomically disadvantaged populations.",
            "2. **Study Context or Setting:** The study is conducted in medical centers across Europe and the United States but does not examine healthcare access disparities or focus on underserved settings such as rural communities or low-income populations.",
            "3. **Examination of Discrimination or Oppression:** The study does not investigate systemic discrimination, racial disparities, or socioeconomic factors affecting kidney transplant outcomes. There is no discussion of inequities in transplant accessibility.",
            "4. **Research Methods or Equity Focus:** The study focuses on validating a risk prediction model (iBox system) for kidney allograft failure. While clinically valuable, it does not incorporate health equity-focused methodologies such as community-based participatory research or culturally tailored interventions.",
            "5. **Additional Considerations:**",
            "    - **Age of participants is not a criterion for determining health equity scholarship:**",
            "        - The study focuses on pediatric patients, but age alone is not sufficient to qualify a study as health equity scholarship.",
            "        - A study must explicitly address disparities or inequities in healthcare access, treatment, or outcomes for marginalized age groups.",
            "        - This study evaluates transplant risk but does not analyze differences in access or outcomes across age groups in an equity-focused manner.",
            "    - **Physical health interventions targeting specific health conditions are not considered part of health equity unless they address systemic inequities or disparities:**",
            "        - The study examines kidney allograft failure, a clinical issue, but does not explore disparities in access to transplantation, follow-up care, or socioeconomic barriers to receiving transplants.",
            "        - Health equity research must go beyond medical risk prediction and address broader social determinants of health, which this study does not do.",
            "6. **Final Evaluation:** The study does not meet the criteria for health equity scholarship, as it does not examine structural barriers, healthcare inequities, or access disparities. Therefore, it does not qualify as health equity scholarship."
        ],
        "final_answer": "No"
    },
    {
        "abstract": "Purpose: Kidney failure is a rare but serious late effect following treatment for childhood cancer. We developed a model using demographic and treatment characteristics to predict individual risk of kidney failure among 5-year survivors of childhood cancer. Methods: Five-year survivors from the Childhood Cancer Survivor Study (CCSS) without history of kidney failure (n = 25,483) were assessed for subsequent kidney failure (ie, dialysis, kidney transplantation, or kidney-related death) by age 40 years. Outcomes were identified by self-report and linkage with the Organ Procurement and Transplantation Network and the National Death Index. A sibling cohort (n = 5,045) served as a comparator. Piecewise exponential models accounting for race/ethnicity, age at diagnosis, nephrectomy, chemotherapy, radiotherapy, congenital genitourinary anomalies, and early-onset hypertension estimated the relationships between potential predictors and kidney failure, using area under the curve (AUC) and concordance (C) statistic to evaluate predictive power. Regression coefficient estimates were converted to integer risk scores. The St Jude Lifetime Cohort Study and the National Wilms Tumor Study served as validation cohorts. Results: Among CCSS survivors, 204 developed late kidney failure. Prediction models achieved an AUC of 0.65-0.67 and a C-statistic of 0.68-0.69 for kidney failure by age 40 years. Validation cohort AUC and C-statistics were 0.88/0.88 for the St Jude Lifetime Cohort Study (n = 8) and 0.67/0.64 for the National Wilms Tumor Study (n = 91). Risk scores were collapsed to form statistically distinct low- (n = 17,762), moderate- (n = 3,784), and high-risk (n = 716) groups, corresponding to cumulative incidences in CCSS of kidney failure by age 40 years of 0.6% (95% CI, 0.4 to 0.7), 2.1% (95% CI, 1.5 to 2.9), and 7.5% (95% CI, 4.3 to 11.6), respectively, compared with 0.2% (95% CI, 0.1 to 0.5) among siblings. Conclusion: Prediction models accurately identify childhood cancer survivors at low, moderate, and high risk for late kidney failure and may inform screening and interventional strategies.",
        "reasoning": [
            "1. **Focus on Disenfranchised Individuals/Populations:** The study focuses on childhood cancer survivors, a medically vulnerable population that may face long-term health consequences. It also considers race/ethnicity as a factor in risk prediction, acknowledging potential disparities in kidney failure outcomes.",
            "2. **Study Context or Setting:** The study uses data from the Childhood Cancer Survivor Study (CCSS), which includes a large, diverse population of childhood cancer survivors. It also incorporates validation cohorts, ensuring that findings are relevant across multiple settings, including historically underserved populations.",
            "3. **Examination of Discrimination or Oppression:** While the study does not explicitly analyze discrimination, it does assess risk factors such as race/ethnicity and insurance-related variables that may reflect systemic disparities in healthcare access and long-term outcomes for childhood cancer survivors.",
            "4. **Research Methods or Equity Focus:** The study employs statistical models to predict kidney failure risk and identifies high-risk groups, which may help improve screening and intervention strategies for those disproportionately affected. By incorporating demographic and socioeconomic factors, the research provides insights into health disparities and potential inequities in long-term cancer survivorship care.",
            "5. **Additional Considerations:**",
            "    - **Age of participants is not a criterion for determining health equity scholarship:**",
            "        - The study focuses on childhood cancer survivors, but age alone does not determine health equity status.",
            "        - However, the study examines the **long-term** effects of cancer treatments, which may be relevant for understanding disparities in survivorship care.",
            "    - **Physical health interventions targeting specific health conditions are not considered part of health equity unless they address systemic inequities or disparities:**",
            "        - The study does more than just predict kidney failure; it also examines risk factors tied to socioeconomic status and race/ethnicity.",
            "        - By identifying vulnerable groups at higher risk of kidney failure, the study helps address disparities in post-cancer healthcare access and outcomes.",
            "6. **Final Evaluation:** The study meets multiple criteria for health equity scholarship, including its focus on a medically vulnerable population, consideration of race/ethnicity in risk assessment, and potential contributions to addressing disparities in survivorship care. Therefore, it qualifies as health equity scholarship."
        ],
        "final_answer": "Yes"
    },
    {
        "abstract": "Herpes simplex virus type 2 (HSV-2) is common globally and contributes significantly to the risk of acquiring HIV-1, yet these two sexually transmitted infections have not been sufficiently characterized for sexual and gender minorities (SGM) across Sub-Saharan Africa. To help fill this gap, we performed a retrospective study using plasma and serum samples from 183 SGM enrolled at the Lagos site of the TRUST/RV368 cohort in Nigeria, assayed them for HSV-2 antibodies with the Kalon ELISA and plasma cytokines and chemokines with Luminex, and correlated the findings with HIV-1 viral loads (VLs) and CD4 counts. We found an overall HSV-2 prevalence of 36.6% (49.5% and 23.9% among SGM with and without HIV-1, respectively, p< .001). Moreover, HSV-2-positive status was associated with high circulating concentrations of CCL11 among antiretroviral therapy-treated (p= .031) and untreated (p= .015) participants, and with high concentrations of CCL2 in the untreated group (p= .004), independent of VL. Principal component analysis revealed a strong association of cytokines with HIV-1 VL independent of HSV-2 status. In conclusion, our study finds that HSV-2 prevalence among SGM with HIV-1 is twice as high than HSV-2 prevalence among SGM without HIV-1 in Lagos and suggests that this is associated with higher levels of certain systemic cytokines. Additional work is needed to further characterize the relationship between HSV-2 and HIV-1 in SGM and help develop targeted therapies for coinfected individuals.",
        "reasoning": [
            "1. **Focus on Disenfranchised Individuals/Populations:** The study explicitly focuses on sexual and gender minorities (SGM) in Sub-Saharan Africa, a historically marginalized and underserved population facing significant health disparities in HIV-1 and HSV-2 infections.",
            "2. **Study Context or Setting:** The research is conducted in Lagos, Nigeria, where access to healthcare and treatment for SGM is often limited due to systemic discrimination and social stigma. The study helps characterize the epidemiology of HSV-2 and HIV-1 in an understudied population.",
            "3. **Examination of Discrimination or Oppression:** The study highlights the lack of sufficient characterization of HSV-2 and HIV-1 among SGM in Sub-Saharan Africa, pointing to potential structural inequalities in research, healthcare access, and treatment availability for these populations.",
            "4. **Research Methods or Equity Focus:** The study uses a retrospective analysis of plasma and serum samples to investigate the correlation between HSV-2 and HIV-1 in SGM, providing crucial insights that could inform targeted interventions and therapies to reduce health disparities among this population.",
            "5. **Additional Considerations:**",
            "    - **Age of participants is not a criterion for determining health equity scholarship:**",
            "        - The study includes adult participants but does not focus on age as a determinant of health inequities.",
            "        - The primary focus is on sexual and gender identity as a risk factor for differential health outcomes.",
            "    - **Physical health interventions targeting specific health conditions are not considered part of health equity unless they address systemic inequities or disparities:**",
            "        - This study directly addresses health inequities by examining the burden of HSV-2 and HIV-1 among a marginalized population.",
            "        - The findings highlight disparities in disease prevalence and suggest the need for improved interventions tailored to SGM in Sub-Saharan Africa.",
            "6. **Final Evaluation:** The study meets multiple criteria for health equity scholarship by focusing on a marginalized population (SGM), addressing disparities in HIV-1 and HSV-2 prevalence, and providing insights that could inform targeted healthcare interventions. Therefore, it qualifies as health equity scholarship."
        ],
        "final_answer": "Yes"
    },  
    {
        "abstract": "Introduction: Race and ethnicity are social constructs that are associated with meaningful health inequities. To address health disparities, it is essential to have valid, reliable race and ethnicity data. We compared child race and ethnicity as identified by the parent with that reported in the electronic health record (EHR). Methods: A convenience sample of parents of pediatric emergency department (PED) patients completed a tablet-based questionnaire (February-May 2021). Parents identified their child's race and ethnicity from options within a single category. We used chi-square to compare concordance between child race and ethnicity reported by the parent with that recorded in the EHR. Results: Of 219 approached parents, 206 (94%) completed questionnaires. Race and/or ethnicity were misidentified in the EHR for 56 children (27%). Misidentifications were most common among children whose parents identified them as multiracial (100% vs 15% of children identified as a single race, P < 0.001) or Hispanic (84% vs 17% of non-Hispanic children, P < 0.001), and children whose race and/or ethnicity differed from that of their parent (79% vs 18% of children with the same race and ethnicity as their parent, P < 0.001). Conclusion: In this PED, misidentification of race and ethnicity was common. This study provides the basis for a multifaceted quality improvement effort at our institution. The quality of child race and ethnicity data in the emergency setting warrants further consideration across health equity efforts.",
        "reasoning": [
            "1. **Focus on Disenfranchised Individuals/Populations:** The study explicitly focuses on race and ethnicity as social constructs that are linked to meaningful health inequities. It examines data misclassification in electronic health records (EHR), particularly among Hispanic and multiracial children, both of whom face systemic disparities in healthcare settings.",
            "2. **Study Context or Setting:** The study is conducted in a pediatric emergency department (PED), a critical setting where accurate race and ethnicity data are essential for equitable care. Given that emergency care is often the first point of healthcare access for many underserved populations, improving data accuracy in this setting has broad implications for health equity.",
            "3. **Examination of Discrimination or Oppression:** The study highlights how misidentification of race and ethnicity is more prevalent among children from historically marginalized racial and ethnic groups. This suggests systemic bias in healthcare documentation, which can contribute to inequities in medical treatment, resource allocation, and research data validity.",
            "4. **Research Methods or Equity Focus:** The study uses a data-driven approach to assess discrepancies in EHR-reported race and ethnicity, employing statistical analysis to identify populations most affected by these inaccuracies. By providing a foundation for quality improvement initiatives, the research supports efforts to reduce bias in healthcare data collection.",
            "5. **Additional Considerations:**",
            "    - **Age of participants is not a criterion for determining health equity scholarship:**",
            "        - The study focuses on pediatric patients, but age alone does not determine health equity status.",
            "        - The emphasis on race and ethnicity misclassification aligns with health equity principles rather than age-based disparities.",
            "    - **Physical health interventions targeting specific health conditions are not considered part of health equity unless they address systemic inequities or disparities:**",
            "        - The study does not focus on a specific medical condition but instead examines systemic barriers related to race and ethnicity data accuracy in healthcare settings.",
            "        - The findings are relevant to reducing disparities in healthcare access, outcomes, and research accuracy.",
            "6. **Final Evaluation:** The study meets multiple criteria for health equity scholarship by focusing on racial and ethnic misclassification in healthcare data, highlighting systemic biases, and contributing to quality improvement efforts in pediatric emergency settings. Therefore, it qualifies as health equity scholarship."
        ],
        "final_answer": "Yes"
    }, 
    {
        "abstract": "Background: Residential segregation is an important factor that negatively impacts cancer disparities, yet studies yield mixed results and complicate clear recommendations for policy change and public health intervention. In this study, we examined the relationship between local and Metropolitan Statistical Area (MSA) measures of Black isolation (segregation) and survival among older non-Hispanic (NH) Black women with breast cancer (BC) in the United States. We hypothesized that the influence of local isolation on mortality varies based on MSA isolation-specifically, that high local isolation may be protective in the context of highly segregated MSAs, as ethnic density may offer opportunities for social support and buffer racialized groups from the harmful influences of racism. Methods: Local and MSA measures of isolation were linked by Census Tract (CT) with a SEER-Medicare cohort of 5,231 NH Black women aged 66-90 years with an initial diagnosis of stage I-IV BC in 2007-2013 with follow-up through 2018. Proportional and cause-specific hazards models and estimated marginal means were used to examine the relationship between local and MSA isolation and all-cause and BC-specific mortality, accounting for covariates (age, comorbidities, tumor stage, and hormone receptor status). Findings: Of 2,599 NH Black women who died, 40.0% died from BC. Women experienced increased risk for all-cause mortality when living in either high local (HR = 1.20; CI = 1.08-1.33; p < 0.001) or high MSA isolation (HR = 1.40; CI = 1.17-1.67; p < 0.001). A similar trend existed for BC-specific mortality. Pairwise comparisons for all-cause mortality models showed that high local isolation was hazardous in less isolated MSAs but was not significant in more isolated MSAs. Interpretation: Both local and MSA isolation are independently associated with poorer overall and BC-specific survival for older NH Black women. However, the impact of local isolation on survival appears to depend on the metropolitan area's level of segregation. Specifically, in highly segregated MSAs, living in an area with high local isolation is not significantly associated with poorer survival. While the reasons for this are not ascertained in this study, it is possible that the protective qualities of ethnic density (e.g., social support and buffering from experiences of racism) may have a greater role in more segregated MSAs, serving as a counterpart to the hazardous qualities of local isolation. More research is needed to fully understand these complex relationships. Funding: National Cancer Institute.",
        "reasoning": [
            "1. **Focus on Disenfranchised Individuals/Populations:** The study focuses on non-Hispanic (NH) Black women with breast cancer, a group that has historically faced significant healthcare disparities. By investigating the impact of residential segregation on cancer survival, the study directly addresses a critical factor contributing to racial health inequities.",
            "2. **Study Context or Setting:** The study examines segregation at both local and metropolitan levels, using Census Tract data and SEER-Medicare cohort information. The research is conducted in the United States, where systemic racial disparities in healthcare access and outcomes persist, particularly for Black women diagnosed with breast cancer.",
            "3. **Examination of Discrimination or Oppression:** The study investigates the structural impact of segregation on breast cancer mortality, highlighting how isolation can negatively affect health outcomes. It explores how high ethnic density in highly segregated MSAs may offer protective effects against racism-related stressors, a key equity-related consideration.",
            "4. **Research Methods or Equity Focus:** The study uses advanced statistical models (proportional hazards and cause-specific models) to analyze the relationship between segregation and survival rates. It accounts for important covariates such as age, comorbidities, tumor stage, and hormone receptor status, ensuring a nuanced understanding of disparities in breast cancer outcomes.",
            "5. **Additional Considerations:**",
            "    - **Age of participants is not a criterion for determining health equity scholarship:**",
            "        - The study focuses on older NH Black women, but age alone does not determine health equity status.",
            "        - The key equity factor is the investigation of how segregation impacts survival disparities.",
            "    - **Physical health interventions targeting specific health conditions are not considered part of health equity unless they address systemic inequities or disparities:**",
            "        - This study does not just examine breast cancer outcomes; it directly ties these outcomes to systemic factors such as racial segregation and isolation.",
            "        - The findings provide insights into how structural racism and neighborhood composition influence survival, highlighting actionable disparities in cancer care and public health.",
            "6. **Final Evaluation:** The study meets multiple criteria for health equity scholarship by focusing on racial disparities in breast cancer survival, analyzing the impact of segregation as a structural determinant of health, and contributing to policy and public health discussions on reducing racial disparities. Therefore, it qualifies as health equity scholarship."
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
    input_file = "/projects/ouzuner/fahmed34/Equity/data/validation_data.csv"
    output_file = "/projects/ouzuner/fahmed34/Equity/results_2/llama31_cotvalran_002.csv"
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
