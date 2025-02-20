import requests
from mistralai import Mistral
from openai import OpenAI, AzureOpenAI

import google.generativeai as genai
import re

text = "Ini adalah contoh<br>teks dengan<br>break."
cleaned_text = re.sub(r"<br>", "", text)
print(cleaned_text)

# Function to create a prompt to generate mitigating controls
def create_mitigations_prompt(threats):
    prompt = f"""
Act as a cyber security expert with over 20 years of experience using the STRIDE threat modeling methodology. You are certified in threat modeling and have specialized experience in the banking industry. Your task is to provide tailored mitigation strategies for identified threats within the threat model. Please respond in Indonesian, but keep all cyber security terminology (e.g., "intercept," "exploitability," "fraudster") in English. Use more advanced technical language to make the responses precise and effective for a professional audience.after that make DFD Diagram Flow based on threat 
dont give recommendation that existing : SAST, DAST, VULNERABILITY ASSESSMENT, PENTEST, CDN, IDS, IPS, FIREWALL, WAF, SIEM,  ANTI DDOS and remove </li></ul> <br>
Your output should be structured as a Markdown table with the following columns:

Column A: Threat Type with STRIDE (spoofing, tampering, repudiation, informastion disclosure, denial of service, elevation privillege ) 

Column B: Scenario

Column C: Suggested Mitigation(s) – for each mitigation, use the following order use bahasa indonesia dan jangan ganti kata serapan nya


Each mitigation should refer to OWASP, ISO27001, MITRE ATT&CK, and ASVS  related to this project. dont use CIAAN, USE STRIDE
example output format :

Please present the entire table in Markdown format with no character limit.
        
Below is the list of identified threats:
{threats}

YOUR RESPONSE (do not wrap in a code block):
"""
    return prompt


# Function to get mitigations from the GPT response.
def get_mitigations(api_key, model_name, prompt):
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model = model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides threat mitigation strategies in Markdown format."},
            {"role": "user", "content": prompt}
        ]
    )

    # Access the content directly as the response will be in text format
    mitigations = response.choices[0].message.content

    return mitigations


# Function to get mitigations from the Azure OpenAI response.
def get_mitigations_azure(azure_api_endpoint, azure_api_key, azure_api_version, azure_deployment_name, prompt):
    client = AzureOpenAI(
        azure_endpoint = azure_api_endpoint,
        api_key = azure_api_key,
        api_version = azure_api_version,
    )

    response = client.chat.completions.create(
        model = azure_deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides threat mitigation strategies in Markdown format."},
            {"role": "user", "content": prompt}
        ]
    )

    # Access the content directly as the response will be in text format
    mitigations = response.choices[0].message.content

    return mitigations

# Function to get mitigations from the Google model's response.
def get_mitigations_google(google_api_key, google_model, prompt):
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel(
        google_model,
        system_instruction="You are a helpful assistant that provides threat mitigation strategies in Markdown format.",
    )
    response = model.generate_content(prompt)
    try:
        # Extract the text content from the 'candidates' attribute
        mitigations = response.candidates[0].content.parts[0].text
        # Replace '\n' with actual newline characters
        mitigations = mitigations.replace('\\n', '\n')
    except (IndexError, AttributeError) as e:
        print(f"Error accessing response content: {str(e)}")
        print("Raw response:")
        print(response)
        return None

    return mitigations

# Function to get mitigations from the Mistral model's response.
def get_mitigations_mistral(mistral_api_key, mistral_model, prompt):
    client = Mistral(api_key=mistral_api_key)

    response = client.chat.complete(
        model = mistral_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides threat mitigation strategies in Markdown format."},
            {"role": "user", "content": prompt}
        ]
    )

    # Access the content directly as the response will be in text format
    mitigations = response.choices[0].message.content

    return mitigations

# Function to get mitigations from Ollama hosted LLM.
def get_mitigations_ollama(ollama_model, prompt):
    
    url = "http://localhost:11434/api/chat"

    data = {
        "model": ollama_model,
        "stream": False,
        "messages": [
            {
                "role": "system", 
                "content": "You are a helpful assistant that provides threat mitigation strategies in Markdown format."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    response = requests.post(url, json=data)

    outer_json = response.json()
    
    # Access the 'content' attribute of the 'message' dictionary
    mitigations = outer_json["message"]["content"]

    return mitigations