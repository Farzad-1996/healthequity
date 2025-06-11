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


file_path = '/projects/ouzuner/fahmed34/Equity/data/75percent_gen.csv'


df = pd.read_csv(file_path)
data= df


prompt_template = """You are an expert in analyzing a study abstract that has already been identified as health equity scholarship. Your goal is to classify it into the highest applicable generation (from 1st to 5th) based on its focus, methods, population, and purpose.

---

### **Instructions for Classification**
Start with **5th Generation** and work downward. Choose the highest generation that applies.

#### **5th Generation – Intersectional Health Equity Intervention Impact**
- Does the study examine the effects of an intervention on individuals with **multiple disenfranchised identities** (such as race + disability, LGBTQ+ + low-income) or between a disenfranchised identity and an under-served or under-resourced context (such as a racial minority population in a rural setting)?
- Does it assess **overlapping systemic barriers** and how these combined factors impact health outcomes, health behaviors, healthcare access, or quality?
- Does the study evaluate how the intervention influences different **subgroups with other disenfranchised identities** or in **under-served or under-resourced contexts**?

**If YES, Final Answer:** `"5th Generation"`  
**If NO, proceed to 4th Generation.**

---

#### **4th Generation – Intervention Impact on Health Outcomes**
- Does the study evaluate the effects of an **intervention specifically tailored** to disenfranchised populations or people in **under-served or under-resourced contexts**?
- Does the study measure the impact of the intervention on **health outcomes, health behaviors, healthcare access, or quality of care**?

**If YES, Final Answer:** `"4th Generation"`  
**If NO, proceed to 3rd Generation.**

---

#### **3rd Generation – Intervention Development & Feasibility**
- Does the study describe the **development of a new or substantively modified intervention** specifically tailored to a **disenfranchised population** or **under-served or under-resourced context**?
- Is the intervention designed to **address barriers, improve access, or promote equitable healthcare** within these populations or contexts?
- Does the study assess **feasibility, acceptability, cultural appropriateness, or implementation factors** without measuring health outcomes?

**If YES, Final Answer:** `"3rd Generation"`  
**If NO, proceed to 2nd Generation.**

---

#### **2nd Generation – Understanding Experiences & Mechanisms**
The study must focus on:
- **People with disenfranchised identities** (e.g., BIPOC, LGBTQ+, low-income, disabled individuals, rural populations), OR
- **People in an under-served or under-resourced context**

**AND at least one of the following must be true:**  
- Exploring **lived experiences, perspectives, or challenges** in accessing healthcare or managing health conditions.  
- Examining **barriers, facilitators, or mechanisms** behind health disparities.  
- Assessing **cultural, structural, or systemic barriers** without proposing or evaluating an intervention.  
- Exploring the impact of an **existing or standard intervention** on a disenfranchised population or in an under-served or under-resourced context **without modification or tailoring**.  
- Using **qualitative research or mixed methods** to understand patient or provider experiences.  
- Studying **experiences of providers** delivering care to these populations.

**If YES, Final Answer:** `"2nd Generation"`  
**If NO, proceed to 1st Generation.**

---

#### **1st Generation – Exploration of Inequities**
To qualify, the study must meet **both** of the following conditions:

1. **Identification of Health Inequities:**  
   - Examines **differences in health behaviors, health outcomes, healthcare, or other health-related factors** between disenfranchised and non-disenfranchised groups, OR  
   - Examines **disparities** based on race, ethnicity, gender, socioeconomic status, disability, or other marginalized identities.  

2. **Non-Interventional Nature:**  
   - The study is **observational, descriptive, or analytical** without attempting to **modify** or **address inequities**.  
   - Lacks **intervention development, evaluation, or tailored modifications** for disenfranchised populations.

**If YES, Final Answer:** `"1st Generation"`

---

### **Important Instructions:**
- **Always select the highest generation applicable** (5th through 1st).  
- **Only provide the final generation classification** as the output (e.g., `"3rd Generation"`).

### **Instructions for Output:**
1. **Your response must start with:** `Final Answer: X Generation`
2. **Then, provide the reasoning behind the classification.**
3. **Do not include any other text, explanations, or disclaimers outside of the Final Answer and reasoning.**  

---

### **Examples of Reasoning Process:**

"""

# Adding the five examples
examples = [
    {
        "abstract": "Purpose: Kidney failure is a rare but serious late effect following treatment for childhood cancer. We developed a model using demographic and treatment characteristics to predict individual risk of kidney failure among 5-year survivors of childhood cancer. Methods: Five-year survivors from the Childhood Cancer Survivor Study (CCSS) without history of kidney failure (n = 25,483) were assessed for subsequent kidney failure (ie, dialysis, kidney transplantation, or kidney-related death) by age 40 years. Outcomes were identified by self-report and linkage with the Organ Procurement and Transplantation Network and the National Death Index. A sibling cohort (n = 5,045) served as a comparator. Piecewise exponential models accounting for race/ethnicity, age at diagnosis, nephrectomy, chemotherapy, radiotherapy, congenital genitourinary anomalies, and early-onset hypertension estimated the relationships between potential predictors and kidney failure, using area under the curve (AUC) and concordance (C) statistic to evaluate predictive power. Regression coefficient estimates were converted to integer risk scores. The St Jude Lifetime Cohort Study and the National Wilms Tumor Study served as validation cohorts. Results: Among CCSS survivors, 204 developed late kidney failure. Prediction models achieved an AUC of 0.65-0.67 and a C-statistic of 0.68-0.69 for kidney failure by age 40 years. Validation cohort AUC and C-statistics were 0.88/0.88 for the St Jude Lifetime Cohort Study (n = 8) and 0.67/0.64 for the National Wilms Tumor Study (n = 91). Risk scores were collapsed to form statistically distinct low- (n = 17,762), moderate- (n = 3,784), and high-risk (n = 716) groups, corresponding to cumulative incidences in CCSS of kidney failure by age 40 years of 0.6% (95% CI, 0.4 to 0.7), 2.1% (95% CI, 1.5 to 2.9), and 7.5% (95% CI, 4.3 to 11.6), respectively, compared with 0.2% (95% CI, 0.1 to 0.5) among siblings. Conclusion: Prediction models accurately identify childhood cancer survivors at low, moderate, and high risk for late kidney failure and may inform screening and interventional strategies.",
        "reasoning": [
            "5th Generation: The study does not examine an intervention on individuals with multiple disenfranchised identities or overlapping systemic barriers. **NO**",
            "4th Generation: The study does not evaluate the impact of an intervention tailored to disenfranchised populations on health outcomes. **NO**",
            "3rd Generation: The study does not describe the development or feasibility of a tailored intervention. **NO**",
            "2nd Generation: The study does not explore lived experiences, mechanisms of disparities, or barriers in disenfranchised populations. **NO**",
            "1st Generation: The study examines differences in health outcomes (kidney failure risk) among a medically vulnerable population (childhood cancer survivors). It considers demographic disparities, including race/ethnicity. It is observational and does not attempt to modify inequities."
        ],
        "final_answer": "1st Generation"
    },
    {
        "abstract": "The National Collegiate Athletic Association (NCAA) Summit on Gender Identity and Student-Athlete Participation was convened to identify institutional/athletic department strategies that may support the well-being of trans and gender nonconforming (TGNC) collegiate student-athletes in the USA. The Summit's purview did not include policy-level changes to eligibility rules. A modified Delphi consensus process was used to identify strategies for supporting collegiate TGNC student-athlete well-being. Key steps included an exploration phase (learning, generating ideas), and an evaluation phase (rating ideas in terms of their utility and feasibility). Summit participants (n=60) included individuals meeting at least one of the following criteria: current or former TGNC athlete, academic or healthcare professional with topical expertise, collegiate athletics stakeholder who would be involved in implementing potential strategies, representative from leading sports medicine organisation, or representative from relevant NCAA membership committee. Summit participants identified strategies in the following domains: healthcare practices (patient-centred care and culturally sensitive care); education for all stakeholders involved in athletics; and administration (inclusive language, quality improvement processes). Summit participants also proposed ways that the NCAA, through its existing committee and governance structures, could help support the well-being of TGNC athletes. NCAA-focused concepts were in the following domains: policy making processes; eligibility and transfer processes; resource development and dissemination; and visibility and support for TGNC athletes. The strategies developed represent important and relevant approaches that member institutions, athletic departments, NCAA committees, governance bodies and other stakeholders might consider in their efforts to support TGNC student-athlete well-being.",
        "reasoning": [
            "5th Generation: The study does not examine the effects of an intervention on individuals with multiple disenfranchised identities or assess overlapping systemic barriers impacting health outcomes. **NO**",
            "4th Generation: The study does not evaluate the effects of a tailored intervention on health outcomes. **NO**",
            "3rd Generation: The study does not describe the development or feasibility of a specific intervention tailored to a disenfranchised population. **NO**",
            "2nd Generation: The study focuses on trans and gender nonconforming (TGNC) student-athletes, a disenfranchised population. It uses qualitative methods (Delphi consensus) to understand their experiences, needs, and strategies to support well-being. It explores barriers, supports, and mechanisms related to TGNC athlete care without delivering an intervention.",
            "1st Generation: The study does not compare differences in health outcomes or experiences between TGNC and non-TGNC populations. **NO**"
        ],
        "final_answer": "2nd Generation"
    },  
    {
        "abstract": "Objective: Children with low income and minority race and ethnicity have worse hospital outcomes due partly to systemic and interpersonal racism causing communication and system barriers. We tested the feasibility and acceptability of a novel inpatient communication-focused navigation program. Methods: Multilingual design workshops with parents, providers, and staff created the Family Bridge Program. Delivered by a trained navigator, it included 1) hospital orientation; 2) social needs screening and response; 3) communication preference assessment; 4) communication coaching; 5) emotional support; and 6) a post-discharge phone call. We enrolled families of hospitalized children with public or no insurance, minority race or ethnicity, and preferred language of English, Spanish, or Somali in a single-arm trial. We surveyed parents at enrollment and 2 to 4 weeks post-discharge, and providers 2 to 3 days post-discharge. Survey measures were analyzed with paired t tests. Results: Of 60 families enrolled, 57 (95%) completed the follow-up survey. Most parents were born outside the United States (60%) with a high school degree or less (60%). Also, 63% preferred English, 33% Spanish, and 3% Somali. The program was feasible: families received an average of 5.3 of 6 components; all received >2. Most caregivers (92%) and providers (81% [30/37]) were \"very satisfied.\" Parent-reported system navigation improved from enrollment to follow-up (+8.2 [95% confidence interval 2.9, 13.6], P = .003; scale 0-100). Spanish-speaking parents reported decreased skills-related barriers (-18.4 [95% confidence interval -1.8, -34.9], P = .03; scale 0-100). Conclusions: The Family Bridge Program was feasible, acceptable, and may have potential for overcoming barriers for hospitalized children at risk for disparities.",
        "reasoning": [
            "5th Generation: The study does not examine the effects of an intervention on individuals with multiple disenfranchised identities or assess overlapping systemic barriers impacting health outcomes. **NO**",
            "4th Generation: Although the study delivers an intervention tailored to disenfranchised populations (families with low income, minority race, and preferred languages other than English), it does not measure health outcomes or health care quality outcomes. **NO**",
            "3rd Generation: The study describes and tests the feasibility and acceptability of a new intervention (the Family Bridge Program) specifically designed for disenfranchised populations. It addresses barriers in communication within the hospital setting but focuses on feasibility and acceptability rather than health outcomes.",
            "2nd Generation: While the study explores experiences and barriers of disenfranchised populations, the primary aim is testing an intervention, which makes it more appropriately classified as 3rd Generation.",
            "1st Generation: The study does not simply observe inequities but actively develops and evaluates an intervention. **NO**"
        ],
        "final_answer": "3rd Generation"
    },
    {
        "abstract": "Children from minoritized/socioeconomically deprived backgrounds suffer disproportionately high rates of uninsurance and graft failure/death after liver transplant. Medicaid expansion was developed to expand access to public insurance. Our objective was to characterize the impact of Medicaid expansion policies on long-term graft/patient survival after pediatric liver transplantation. All pediatric patients (<19 years) who received a liver transplant between January 1, 2005, and December 31, 2020 in the US were identified in the Scientific Registry of Transplant Recipients (N = 8489). Medicaid expansion was modeled as a time-varying exposure based on transplant and expansion dates. We used Cox proportional hazards models to evaluate the impact of Medicaid expansion on a composite outcome of graft failure/death over 10 years. As a sensitivity analysis, we conducted an intention-to-treat analysis from time of waitlisting to death (N = 1 1901). In multivariable analysis, Medicaid expansion was associated with a 30% decreased hazard of graft failure/death (hazard ratio, 0.70; 95% confidence interval, 0.62, 0.79; P < .001) after adjusting for Black race, public insurance, neighborhood deprivation, and living in a primary care shortage area. In intention-to-treat analyses, Medicaid expansion was associated with a 72% decreased hazard of patient death (hazard ratio, 0.28; 95% confidence interval, 0.23-0.35; P < .001). Policies that enable broader health insurance access may help improve outcomes and reduce disparities for children undergoing liver transplantation.",
        "reasoning": [
            "5th Generation: The study does not examine the effects of an intervention on individuals with multiple disenfranchised identities or assess how overlapping systemic barriers impact health outcomes. While it adjusts for individual factors like race and socioeconomic status, it does not evaluate intersectional impacts on subgroups. **NO**",
            "4th Generation: The study evaluates the impact of a policy intervention (Medicaid expansion) specifically targeting access to health care for socioeconomically deprived populations, including minoritized children. It measures the impact on health outcomes (graft failure and death), making it a clear example of 4th Generation health equity scholarship.",
            "3rd Generation: Although the study involves a policy intervention, it does not focus on the development or feasibility of a new or modified intervention. **NO**",
            "2nd Generation: The study does not primarily explore lived experiences, mechanisms, or qualitative factors. **NO**",
            "1st Generation: The study is not purely observational of inequities without an intervention. **NO**"
        ],
        "final_answer": "4th Generation"
    }
]

# Append examples to the prompt
for idx, example in enumerate(examples, 1):
    prompt_template += f"\n### **Example {idx}:**\n**Abstract:**\n{example['abstract']}\n\n"
    prompt_template += "**Step-by-Step Reasoning:**\n"
    for step in example["reasoning"]:
        prompt_template += f"- {step}\n"
    prompt_template += f"\n**Final Answer:** `{example['final_answer']}`\n"

# Append user abstract classification request
prompt_template += """\n
### **Now, classify the following abstract:**

**Abstract:**  
"{}"
"""


# Define function for zero-shot classification
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
data['Generated Response'] = predictions
data.to_csv("/projects/ouzuner/fahmed34/Equity/results/gen_classified_data_gpt4o_nshotCOT-001.csv", index=False)

print("Predictions saved to classified_data.csv")
