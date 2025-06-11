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


file_path = '/projects/ouzuner/fahmed34/Equity/results/predicted_no_gpt4osecondprompt_001.csv'


df = pd.read_csv(file_path)
data= df

# Define the prompt template
prompt_template = """You are an expert in analyzing study abstracts to determine if they qualify as people-focused health equity scholarship.

Follow these structured steps for reasoning:

    1. **Examination of Discrimination or Oppression:

    - Does the study explicitly explore the impact of racism, sexism, ableism, xenophobia, or other forms of discrimination or oppression?
        * Does it examine strategies to reduce bias, harm, or inequities (e.g., training programs for providers to improve interactions with marginalized populations)?
        * Age of participants alone does not determine health equity scholarship. Additionally, studies focused solely on physical health interventions do not qualify unless they address systemic inequities or disparities.
    
    **Final Evaluation:**

    - If the study meets any of the above criteria, it qualifies as health equity scholarship.
    - If ambiguity or partial focus exists, classify the study as "Yes" to avoid missing relevant work.
    - Final Answer: Provide either Yes or No, indicating whether the study qualifies as health equity scholarship.

    ### **Instructions for Output:**
    1. **Your response must start with:** `Final Answer: Yes` or `Final Answer: No`
    2. **Then, provide the reasoning behind the classification.**
    3. **Do not include any other text, explanations, or disclaimers outside of the Final Answer and reasoning.**

    **Examples of Reasoning Process:** 

---
         ### **Example 1**
        **Abstract:**
        "Previous research suggests that routine psychosocial care for adolescents with attention-deficit/hyperactivity disorder (ADHD) is an eclectic and individualized mix of diluted evidence-based practices (EBPs) and low-value approaches. This study evaluated the extent to which a community-delivered EBP and usual care (UC) for adolescents with ADHD produce differential changes in theorized behavioral, psychological, and cognitive mechanisms of ADHD. A randomized community-based trial was conducted with double randomization of adolescent and community therapists to EBP delivery supports (Supporting Teens' Autonomy Daily [STAND]) versus UC delivery. Participants were 278 culturally diverse adolescents (ages 11-17) with ADHD and caregivers. Mechanistic outcomes were measured at baseline, post-treatment, and follow-up using parent-rated, observational, and task-based measures. Results using linear mixed models indicated that UC demonstrated superior effects on parent-rated and task-based executive functioning relative to STAND. However, STAND demonstrated superior effects on adolescent motivation and reducing parental intrusiveness relative to UC when it was delivered by licensed therapists. Mechanisms of community-delivered STAND and UC appear to differ. UC potency may occur through improved executive functioning, whereas STAND potency may occur through improved teen motivation and reducing low-value parenting practices. However, when delivered by unlicensed, community-based therapists, STAND did not enact proposed mechanisms. Future adaptations of community-delivered EBPs for ADHD should increase supports for unlicensed therapists, who comprise the majority of the community mental health workforce.",
        **Reasoning:** 
            "1. **Examination of Discrimination or Oppression:** The study highlights disparities in community mental health treatment by examining differences in therapy effectiveness when delivered by licensed vs. unlicensed therapists. The findings suggest that unlicensed community-based therapists, who make up the majority of the workforce, may not have adequate training or support, leading to lower effectiveness of evidence-based interventions like STAND.",
            "The study also touches on **systemic inequities** in access to quality ADHD care, as **culturally diverse adolescents** receiving care from unlicensed therapists may experience less effective treatment. This reflects broader disparities in the mental healthcare system, particularly for families who rely on community-based services.",
            "Additionally, the study suggests a **need for structural improvements**, including better support for unlicensed therapists, which aligns with efforts to reduce harm and inequities in mental healthcare. By identifying barriers that disproportionately impact underserved groups, the study contributes to discussions on reducing disparities in ADHD treatment.",
            "**Final Evaluation:** The study qualifies as health equity scholarship because it examines systemic barriers in community mental health, highlights inequities in therapist training and effectiveness, and suggests structural improvements to reduce disparities in ADHD care."
        
        **Final Answer:** "Yes"

        ### **Example 2**
        **Abstract:** 
        "Objective: Children with low income and minority race and ethnicity have worse hospital outcomes due partly to systemic and interpersonal racism causing communication and system barriers. We tested the feasibility and acceptability of a novel inpatient communication-focused navigation program. Methods: Multilingual design workshops with parents, providers, and staff created the Family Bridge Program. Delivered by a trained navigator, it included 1) hospital orientation; 2) social needs screening and response; 3) communication preference assessment; 4) communication coaching; 5) emotional support; and 6) a post-discharge phone call. We enrolled families of hospitalized children with public or no insurance, minority race or ethnicity, and preferred language of English, Spanish, or Somali in a single-arm trial. We surveyed parents at enrollment and 2 to 4 weeks post-discharge, and providers 2 to 3 days post-discharge. Survey measures were analyzed with paired t tests. Results: Of 60 families enrolled, 57 (95%) completed the follow-up survey. Most parents were born outside the United States (60%) with a high school degree or less (60%). Also, 63% preferred English, 33% Spanish, and 3% Somali. The program was feasible: families received an average of 5.3 of 6 components; all received >2. Most caregivers (92%) and providers (81% [30/37]) were 'very satisfied.' Parent-reported system navigation improved from enrollment to follow-up (+8.2 [95% confidence interval 2.9, 13.6], P = .003; scale 0-100). Spanish-speaking parents reported decreased skills-related barriers (-18.4 [95% confidence interval -1.8, -34.9], P = .03; scale 0-100). Conclusions: The Family Bridge Program was feasible, acceptable, and may have potential for overcoming barriers for hospitalized children at risk for disparities.",
        **Reasoning:**  
            "1. **Examination of Discrimination or Oppression:** The study directly examines systemic and interpersonal racism as a contributing factor to worse hospital outcomes for children from low-income and minority racial and ethnic backgrounds. It acknowledges that structural discrimination leads to communication barriers and disparities in hospital care.",
            "The Family Bridge Program is designed to address these inequities by improving communication, social support, and system navigation for families with limited English proficiency and lower income. By focusing on marginalized communities, including Spanish and Somali speakers, the study highlights how linguistic and economic barriers intersect with systemic racism in healthcare settings.",
            "The study also evaluates whether a trained navigator can mitigate these disparities through interventions such as hospital orientation, social needs screening, and post-discharge follow-up. The significant improvement in navigation and reduction in skill-related barriers among Spanish-speaking parents suggests that targeted strategies can reduce inequities in hospital care.",
            "**Final Evaluation:** The study qualifies as health equity scholarship under Prompt 3 because it explicitly examines the impact of systemic racism in hospital care and evaluates a structured intervention aimed at reducing bias, harm, and disparities in healthcare access and outcomes."
        
        **Final Answer:** "Yes"
    
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
                
                delay = 10  
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
batch_size = 10  
predictions = []

for i in range(0, len(data), batch_size):
    batch_texts = data['Abstract'][i:i+batch_size].tolist()
    batch_predictions = zero_shot_classify_batch(batch_texts)
    predictions.extend(batch_predictions)
    time.sleep(1)  


if len(predictions) < len(data):
    predictions.extend(["Error"] * (len(data) - len(predictions)))


data['prediction'] = predictions
data.to_csv("/projects/ouzuner/fahmed34/Equity/results/classified_data_gpt4o_thirdpromt-001.csv", index=False)

print("Predictions saved to classified_data.csv")
