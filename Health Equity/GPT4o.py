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


file_path = '/projects/ouzuner/fahmed34/Equity/data/cleaned_random_sample_75_percent.csv'


df = pd.read_csv(file_path)
data= df


prompt_template = """You are an expert in analyzing study abstracts to determine if they qualify as people-focused health equity scholarship.

Please analyze the abstract by following these structured steps:

1. **Focus on Disenfranchised Individuals/Populations**:
   - Does the study explicitly include or focus on disenfranchised populations? These include:
     - BIPOC (Black, Indigenous, People of Color), including Hispanic/Latino/a/x individuals.
     - LGBTQ+, gender diverse, or transgender individuals.
     - Individuals with disabilities or intellectual/developmental delays.
     - Socioeconomically disadvantaged groups (e.g., poverty, food insecurity, lack of healthcare access).
     - Individuals who speak a language other than English.
     - Populations in underserved settings (e.g., rural areas, Federally Qualified Health Centers, community hospitals, Title 1 schools, or low-and-middle-income countries).
     - Groups disproportionately affected by specific conditions (e.g., sickle cell disease, malaria, HIV).
     - Women (to consider sex-based disparities).
   - Does the study explore health disparities, inequities, or differences in health outcomes for these groups?

2. **Study Context or Setting**:
   - Is the study conducted in underserved or resource-limited settings (e.g., rural areas, community hospitals, or the Global South)?
   - Does it examine health conditions disproportionately affecting disenfranchised populations?

3. **Examination of Discrimination or Oppression**:
   - Does the study investigate the impact of racism, sexism, ableism, xenophobia, or other systemic biases?
   - Does it explore interventions aimed at reducing discrimination, bias, or inequities (e.g., training programs to improve provider interactions with marginalized populations)?

4. **Research Methods or Equity Focus**:
   - Does the study use equity-centered methodologies, such as:
     - Community-based participatory research.
     - Culturally or linguistically adapted tools (e.g., non-English surveys).
     - Lived experiences in underserved settings.

5. **Additional Considerations**:
   - Age of participants alone does not determine health equity scholarship.
   - Physical health interventions are only considered if they address systemic inequities.

6. **Final Evaluation**:
   - If the study meets any of the above criteria, classify it as "Health Equity Scholarship".
   - If there is ambiguity, classify it as "Health Equity Scholarship" to ensure inclusivity.

### **Instructions for Output:**
1. **Your response must start with:** `Final Answer: Yes` or `Final Answer: No`
2. **Then, provide the reasoning behind the classification.**
3. **Do not include any other text, explanations, or disclaimers outside of the Final Answer and reasoning.**

**Examples of Reasoning Process:**

### **Example 1**
**Abstract:**  
"Predicting long-term kidney allograft failure is an unmet need for clinical care and clinical trial optimization in children. We aimed to validate a kidney allograft failure risk prediction system in a large international cohort of pediatric kidney transplant recipients..."
  
**Reasoning:**  
1. **Focus on Disenfranchised Individuals/Populations:** Pediatric patients are a medically vulnerable group, but the study does not explicitly focus on marginalized populations (e.g., BIPOC, LGBTQ+).  
2. **Study Context or Setting:** Conducted in Europe and the U.S., but does not focus on underserved populations.  
3. **Examination of Discrimination or Oppression:** Does not analyze systemic biases or disparities in healthcare access.  
4. **Research Methods or Equity Focus:** Focuses on clinical risk prediction rather than equity-focused methodologies.  
5. **Final Evaluation:** The study does not meet the criteria for health equity scholarship.  

**Final Answer:** No  

---

### **Example 2**
**Abstract:**  
"Purpose: Kidney failure is a rare but serious late effect following treatment for childhood cancer. We developed a model using demographic and treatment characteristics to predict individual risk of kidney failure among 5-year survivors of childhood cancer..."
  
**Reasoning:**  
1. **Focus on Disenfranchised Individuals/Populations:** The study considers race/ethnicity as a factor in predicting disparities in kidney failure.  
2. **Study Context or Setting:** Uses diverse datasets, including historically underserved groups.  
3. **Examination of Discrimination or Oppression:** Includes race and socioeconomic status, suggesting awareness of health inequities.  
4. **Research Methods or Equity Focus:** Identifies high-risk groups, aiding interventions for at-risk populations.  
5. **Final Evaluation:** The study meets health equity scholarship criteria.  

**Final Answer:** Yes  

---

### **Example 3**
**Abstract:**  
"Herpes simplex virus type 2 (HSV-2) is common globally and contributes significantly to the risk of acquiring HIV-1, yet these two sexually transmitted infections have not been sufficiently characterized for sexual and gender minorities (SGM) across Sub-Saharan Africa..."
  
**Reasoning:**  
1. **Focus on Disenfranchised Individuals/Populations:** The study focuses on sexual and gender minorities (SGM) in Sub-Saharan Africa, a marginalized group.  
2. **Study Context or Setting:** Conducted in Nigeria, where SGM face healthcare discrimination.  
3. **Examination of Discrimination or Oppression:** Highlights disparities in disease prevalence and healthcare access.  
4. **Research Methods or Equity Focus:** Uses epidemiological data to inform interventions for SGM.  
5. **Final Evaluation:** Meets health equity scholarship criteria.  

**Final Answer:** Yes  

---

### **Example 4**
**Abstract:**  
"Introduction: Race and ethnicity are social constructs that are associated with meaningful health inequities. To address health disparities, it is essential to have valid, reliable race and ethnicity data..."
  
**Reasoning:**  
1. **Focus on Disenfranchised Individuals/Populations:** Examines racial and ethnic misclassification in healthcare data, affecting marginalized groups.  
2. **Study Context or Setting:** Conducted in a pediatric emergency department, highlighting disparities in data accuracy.  
3. **Examination of Discrimination or Oppression:** Identifies systemic biases in electronic health records (EHR).  
4. **Research Methods or Equity Focus:** Uses statistical analysis to assess disparities, supporting health equity improvements.  
5. **Final Evaluation:** Meets health equity scholarship criteria.  

**Final Answer:** Yes  

---

### **Example 5**
**Abstract:**  
"Background: Residential segregation is an important factor that negatively impacts cancer disparities, yet studies yield mixed results and complicate clear recommendations for policy change and public health intervention..."
  
**Reasoning:**  
1. **Focus on Disenfranchised Individuals/Populations:** Focuses on non-Hispanic Black women, a historically marginalized group.  
2. **Study Context or Setting:** Investigates how segregation impacts cancer survival, linking structural inequities to health disparities.  
3. **Examination of Discrimination or Oppression:** Highlights systemic racism and isolation’s effects on healthcare outcomes.  
4. **Research Methods or Equity Focus:** Uses statistical modeling to analyze segregation’s role in cancer survival disparities.  
5. **Final Evaluation:** Meets health equity scholarship criteria.  

**Final Answer:** Yes  

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
                    max_tokens=300,
                    temperature=0.0
                )
                prediction = response['choices'][0]['message']['content'].strip()
                predictions.append(prediction)
                
                delay = 10  # Reset delay after a successful call
                break  
            
            except openai.RateLimitError:
                print(f"⚠️ Rate limit reached. Retrying in {delay} seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(delay + random.uniform(1, 3))  # Add slight randomness to avoid collisions
                delay *= 2  # Exponential backoff (10s → 20s → 40s)
            
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
    time.sleep(1)  

# Ensure predictions length matches data length
if len(predictions) < len(data):
    predictions.extend(["Error"] * (len(data) - len(predictions)))

# Add predictions to data and save to CSV
data['prediction'] = predictions
data.to_csv("/projects/ouzuner/fahmed34/Equity/results/classified_data_gpt4o_nshotCOT-002.csv", index=False)

print("Predictions saved to classified_data.csv")
