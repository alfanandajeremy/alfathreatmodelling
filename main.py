#main.py

import base64
import requests
import streamlit as st
import streamlit.components.v1 as components
from github import Github
from collections import defaultdict
import re
import os
from dotenv import load_dotenv

from threat_model import create_threat_model_prompt, get_threat_model, get_threat_model_azure, get_threat_model_google, get_threat_model_mistral, get_threat_model_ollama, json_to_markdown, get_image_analysis, create_image_analysis_prompt
from attack_tree import create_attack_tree_prompt, get_attack_tree, get_attack_tree_azure, get_attack_tree_mistral, get_attack_tree_ollama
from mitigations import create_mitigations_prompt, get_mitigations, get_mitigations_azure, get_mitigations_google, get_mitigations_mistral, get_mitigations_ollama
from test_cases import create_test_cases_prompt, get_test_cases, get_test_cases_azure, get_test_cases_google, get_test_cases_mistral, get_test_cases_ollama
from dread import create_dread_assessment_prompt, get_dread_assessment, get_dread_assessment_azure, get_dread_assessment_google, get_dread_assessment_mistral, get_dread_assessment_ollama, dread_json_to_markdown

# ------------------ Helper Functions ------------------ #

# Function to get user input for the application description and key details
def get_input():
    github_url = st.text_input(
        label="",
        placeholder="",
        key="github_url",
        help="Enter the URL of the GitHub repository you want to analyze.",
    )

    if github_url and github_url != st.session_state.get('last_analyzed_url', ''):
        if 'github_api_key' not in st.session_state or not st.session_state['github_api_key']:
            st.warning("Please enter a GitHub API key to analyze the repository.")
        else:
            with st.spinner('Analyzing GitHub repository...'):
                system_description = analyze_github_repo(github_url)
                st.session_state['github_analysis'] = system_description
                st.session_state['last_analyzed_url'] = github_url
                st.session_state['app_input'] = system_description + "\n\n" + st.session_state.get('app_input', '')

    input_text = st.text_area(
        label="Describe the application to be modelled",
        value=st.session_state.get('app_input', ''),
        placeholder="Enter your application details...",
        height=300,
        key="app_desc",
        help="Please provide a detailed description of the application, including the purpose of the application, the technologies used, and any other relevant information.",
    )

    st.session_state['app_input'] = input_text

    return input_text

def analyze_github_repo(repo_url):
    # Extract owner and repo name from URL
    parts = repo_url.split('/')
    owner = parts[-2]
    repo_name = parts[-1]

    # Initialize PyGithub
    g = Github(st.session_state.get('github_api_key', ''))

    # Get the repository
    repo = g.get_repo(f"{owner}/{repo_name}")

    # Get the default branch
    default_branch = repo.default_branch

    # Get the tree of the default branch
    tree = repo.get_git_tree(default_branch, recursive=True)

    # Analyze files
    file_summaries = defaultdict(list)
    total_chars = 0
    char_limit = 100000  # Adjust this based on your model's token limit
    readme_content = ""

    for file in tree.tree:
        if file.path.lower() == 'readme.md':
            content = repo.get_contents(file.path, ref=default_branch)
            readme_content = base64.b64decode(content.content).decode()
        elif file.type == "blob" and file.path.endswith(('.py', '.js', '.ts', '.html', '.css', '.java', '.go', '.rb')):
            content = repo.get_contents(file.path, ref=default_branch)
            decoded_content = base64.b64decode(content.content).decode()
            
            # Summarize the file content
            summary = summarize_file(file.path, decoded_content)
            file_summaries[file.path.split('.')[-1]].append(summary)
            
            total_chars += len(summary)
            if total_chars > char_limit:
                break

    # Compile the analysis into a system description
    system_description = f"Repository: {repo_url}\n\n"
    
    if readme_content:
        system_description += "README.md Content:\n"
        # Truncate README if it's too long
        if len(readme_content) > 5000:
            system_description += readme_content[:5000] + "...\n(README truncated due to length)\n\n"
        else:
            system_description += readme_content + "\n\n"

    for file_type, summaries in file_summaries.items():
        system_description += f"{file_type.upper()} Files:\n"
        for summary in summaries:
            system_description += summary + "\n"
        system_description += "\n"

    return system_description

def summarize_file(file_path, content):
    # Extract important parts of the file
    imports = re.findall(r'^import .*|^from .* import .*', content, re.MULTILINE)
    functions = re.findall(r'def .*\(.*\):', content)
    classes = re.findall(r'class .*:', content)

    summary = f"File: {file_path}\n"
    if imports:
        summary += "Imports:\n" + "\n".join(imports[:5]) + "\n"  # Limit to first 5 imports
    if functions:
        summary += "Functions:\n" + "\n".join(functions[:5]) + "\n"  # Limit to first 5 functions
    if classes:
        summary += "Classes:\n" + "\n".join(classes[:5]) + "\n"  # Limit to first 5 classes

    return summary

# Function to render Mermaid diagram
def mermaid(code: str, height: int = 500) -> None:
    components.html(
        f"""
        <pre class="mermaid" style="height: {height}px;">
            {code}
        </pre>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        height=height,
    )

def load_env_variables():
    # Try to load from .env file
    if os.path.exists('.env'):
        load_dotenv('.env')
    
    # Load GitHub API key from environment variable
    github_api_key = os.getenv('GITHUB_API_KEY')
    if github_api_key:
        st.session_state['github_api_key'] = github_api_key

    # Load other API keys if needed
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        st.session_state['openai_api_key'] = openai_api_key

    azure_api_key = os.getenv('AZURE_API_KEY')
    if azure_api_key:
        st.session_state['azure_api_key'] = azure_api_key

    azure_api_endpoint = os.getenv('AZURE_API_ENDPOINT')
    if azure_api_endpoint:
        st.session_state['azure_api_endpoint'] = azure_api_endpoint

    azure_deployment_name = os.getenv('AZURE_DEPLOYMENT_NAME')
    if azure_deployment_name:
        st.session_state['azure_deployment_name'] = azure_deployment_name

    google_api_key = os.getenv('GOOGLE_API_KEY')
    if google_api_key:
        st.session_state['google_api_key'] = google_api_key

    mistral_api_key = os.getenv('MISTRAL_API_KEY')
    if mistral_api_key:
        st.session_state['mistral_api_key'] = mistral_api_key

# Call this function at the start of your app
load_env_variables()

# ------------------ Streamlit UI Configuration ------------------ #

st.set_page_config(
    page_title="ALFA THREAT MODELLING AI ",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------ Sidebar ------------------ #

st.sidebar.image("wkwk.png")

# Add instructions on how to use the app to the sidebar
st.sidebar.header("ALFA THREAT MODELLING AI ")

with st.sidebar:
    # Add model selection input field to the sidebar
    model_provider = st.selectbox(
        "Pilih model AI (default GOOGLE AI karna gratis):",
        ["OpenAI API", "Azure OpenAI Service", "Google AI API", "Mistral API", "Ollama"],
        key="model_provider",
        help="Select the model provider you would like to use. This will determine the models available for selection.",
    )

    if model_provider == "OpenAI API":
        st.markdown(
        """
    """
    )
        # Add OpenAI API key input field to the sidebar
        openai_api_key = st.text_input(
            "Enter your OpenAI API key:",
            value=st.session_state.get('openai_api_key', ''),
            type="password",
            help="You can find your OpenAI API key on the [OpenAI dashboard](https://platform.openai.com/account/api-keys).",
        )
        if openai_api_key:
            st.session_state['openai_api_key'] = openai_api_key

        # Add model selection input field to the sidebar
        selected_model = st.selectbox(
            "Select the model you would like to use:",
            ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            key="selected_model",
            help="GPT-4o and GPT-4o mini are OpenAI's latest models and are recommended."
        )

    if model_provider == "Azure OpenAI Service":
        st.markdown(
        """

    """
    )

        # Add Azure OpenAI API key input field to the sidebar
        azure_api_key = st.text_input(
            "Azure OpenAI API key:",
            value=st.session_state.get('azure_api_key', ''),
            type="password",
            help="You can find your Azure OpenAI API key on the [Azure portal](https://portal.azure.com/).",
        )
        if azure_api_key:
            st.session_state['azure_api_key'] = azure_api_key
        
        # Add Azure OpenAI endpoint input field to the sidebar
        azure_api_endpoint = st.text_input(
            "Azure OpenAI endpoint:",
            value=st.session_state.get('azure_api_endpoint', ''),
            help="Example endpoint: https://YOUR_RESOURCE_NAME.openai.azure.com/",
        )
        if azure_api_endpoint:
            st.session_state['azure_api_endpoint'] = azure_api_endpoint

        # Add Azure OpenAI deployment name input field to the sidebar
        azure_deployment_name = st.text_input(
            "Deployment name:",
            value=st.session_state.get('azure_deployment_name', ''),
        )
        if azure_deployment_name:
            st.session_state['azure_deployment_name'] = azure_deployment_name
        
        st.info("Please note that you must use an 1106-preview model deployment.")

        azure_api_version = '2023-12-01-preview' # Update this as needed

        st.write(f"Azure API Version: {azure_api_version}")

    if model_provider == "Google AI API":
        st.markdown(
        """

    """
    )
        # Add OpenAI API key input field to the sidebar
        google_api_key = st.text_input(
            "Enter your Google AI API key:",
            value=st.session_state.get('google_api_key', ''),
            type="password",
            help="You can generate a Google AI API key in the [Google AI Studio](https://makersuite.google.com/app/apikey).",
        )
        if google_api_key:
            st.session_state['google_api_key'] = google_api_key

        # Add model selection input field to the sidebar
        google_model = st.selectbox(
            "Select the model you would like to use:",
            ["gemini-1.5-pro-latest", "gemini-1.5-pro"],
            key="selected_model",
        )

    if model_provider == "Mistral API":
        st.markdown(
        """
  
    """
    )
        # Add OpenAI API key input field to the sidebar
        mistral_api_key = st.text_input(
            "Enter your Mistral API key:",
            value=st.session_state.get('mistral_api_key', ''),
            type="password",
            help="You can generate a Mistral API key in the [Mistral console](https://console.mistral.ai/api-keys/).",
        )
        if mistral_api_key:
            st.session_state['mistral_api_key'] = mistral_api_key

        # Add model selection input field to the sidebar
        mistral_model = st.selectbox(
            "Select the model you would like to use:",
            ["mistral-large-latest", "mistral-small-latest"],
            key="selected_model",
        )

    if model_provider == "Ollama":
        # Make a request to the Ollama API to get the list of available models
        try:
            response = requests.get("http://localhost:11434/api/tags")
            response.raise_for_status() # Raise an exception for 4xx/5xx status codes
        except requests.exceptions.RequestException as e:
            st.error("Ollama endpoint not found, please select a different model provider.")
            response = None
        
        if response:
            data = response.json()
            available_models = [model["name"] for model in data["models"]]
            # Add model selection input field to the sidebar
            ollama_model = st.selectbox(
                "Select the model you would like to use:",
                available_models,
                key="selected_model",
            )



# Add "About" section to the sidebar
st.sidebar.header("About")

with st.sidebar:
    st.markdown(
        "Selamat datang di Alfa Threat Modelling. Aplikasi ini dirancang untuk membantu Anda mengidentifikasi dan mengelola risiko keamanan siber dan menggunakan bantuan powerful gen AI untuk memberikan visibilitas security secara lebih luas dan secara efektif. Selamat menggunakan!"
    )
    st.markdown(
        "Selamat melakukan Assessment kawan!"
    )
    st.markdown("Created by [Jeremy Alfananda](https://www.linkedin.com/in/jeremy-alfananda-5a326ab7/).")
    # Add "Star on GitHub" link to the sidebar
    st.sidebar.markdown(
        "⭐ Developed by Jeremy Alfananda"
    )
    st.markdown("""---""")


# Add "Contoh Prompt yang dapat digunakan" section to the sidebar
st.sidebar.header("Contoh Prompt buat deskripsi")

with st.sidebar:
    st.markdown(
        "dibawah ini contohnya cobain aja. bisa pake bahasa indonesia"
    )
    st.markdown(
        "> A web application that allows users to create, store, and share personal notes. The application is built using the React frontend framework and a Node.js backend with a MongoDB database. Users can sign up for an account and log in using OAuth2 with Google or Facebook. The notes are encrypted at rest and are only accessible by the user who created them. The application also supports real-time collaboration on notes with other users."
    )
    st.markdown("""---""")



# ------------------ Main App UI ------------------ #

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Threat Model", "Attack Tree", "CIAAN dan Mitigations", "DREAD & CIAAN", "Test Cases"])

with tab1:
    st.markdown("""
 membantu mengidentifikasi dan mengevaluasi potensi ancaman keamanan terhadap aplikasi/sistem. Ini menyediakan pendekatan sistematis untuk memahami kerentanan dan vektor serangan yang mungkin terjadi. Gunakan tab ini untuk menghasilkan model ancaman menggunakan metodologi STRIDE.""")
    st.markdown("""---""")
    
    # Two column layout for the main app content
    col1, col2 = st.columns([1, 1])

    # Initialize app_input in the session state if it doesn't exist
    if 'app_input' not in st.session_state:
        st.session_state['app_input'] = ''

    # If model provider is OpenAI API and the model is gpt-4-turbo or gpt-4o
    with col1:
        if model_provider == "OpenAI API" and selected_model in ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini"]:
            uploaded_file = st.file_uploader("Upload architecture diagram", type=["jpg", "jpeg", "png" , "pdf"])

            if uploaded_file is not None:
                if not openai_api_key:
                    st.error("Please enter your OpenAI API key to analyse the image.")
                else:
                    if 'uploaded_file' not in st.session_state or st.session_state.uploaded_file != uploaded_file:
                        st.session_state.uploaded_file = uploaded_file
                        with st.spinner("Analysing the uploaded image..."):
                            def encode_image(uploaded_file):
                                return base64.b64encode(uploaded_file.read()).decode('utf-8')

                            base64_image = encode_image(uploaded_file)

                            image_analysis_prompt = create_image_analysis_prompt()

                            try:
                                image_analysis_output = get_image_analysis(openai_api_key, selected_model, image_analysis_prompt, base64_image)
                                if image_analysis_output and 'choices' in image_analysis_output and image_analysis_output['choices'][0]['message']['content']:
                                    image_analysis_content = image_analysis_output['choices'][0]['message']['content']
                                    st.session_state.image_analysis_content = image_analysis_content
                                    # Update app_input session state
                                    st.session_state['app_input'] = image_analysis_content
                                else:
                                    st.error("Failed to analyze the image. Please check the API key and try again.")
                            except KeyError as e:
                                st.error("Failed to analyze the image. Please check the API key and try again.")
                                print(f"Error: {e}")
                            except Exception as e:
                                st.error("An unexpected error occurred while analyzing the image.")
                                print(f"Error: {e}")

        # Use the get_input() function to get the application description and GitHub URL
        app_input = get_input()
        # Update session state only if the text area content has changed
        if app_input != st.session_state['app_input']:
            st.session_state['app_input'] = app_input

    # Ensure app_input is always up to date in the session state
    app_input = st.session_state['app_input']



        # Create input fields for additional details
    with col2:
            app_type = st.selectbox(
                label="Select the application type",
                options=[
                    "Web application",
                    "Mobile application",
                    "Desktop application",
                    "Cloud application",
                    "IoT application",
                    "Other",
                ],
                key="app_type",
            )

            sensitive_data = st.selectbox(
                label="What is the highest sensitivity level of the data processed by the application?",
                options=[
                    "Top Secret",
                    "Secret",
                    "Confidential",
                    "Restricted",
                    "Unclassified",
                    "None",
                ],
                key="sensitive_data",
            )

        # Create input fields for internet_facing and authentication
            internet_facing = st.selectbox(
                label="Is the application internet-facing?",
                options=["Yes", "No"],
                key="internet_facing",
            )

            authentication = st.multiselect(
                "What authentication methods are supported by the application?",
                ["SSO", "MFA", "OAUTH2", "Basic", "None"],
                key="authentication",
            )



    # ------------------ Threat Model Generation ------------------ #

    # Create a submit button for Threat Modelling
    threat_model_submit_button = st.button(label="Generate Threat Model")

    # If the Generate Threat Model button is clicked and the user has provided an application description
    if threat_model_submit_button and st.session_state.get('app_input'):
        app_input = st.session_state['app_input']  # Retrieve from session state
        # Generate the prompt using the create_prompt function
        threat_model_prompt = create_threat_model_prompt(app_type, authentication, internet_facing, sensitive_data, app_input)

        # Show a spinner while generating the threat model
        with st.spinner("Analysing potential threats..."):
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Call the relevant get_threat_model function with the generated prompt
                    if model_provider == "Azure OpenAI Service":
                        model_output = get_threat_model_azure(azure_api_endpoint, azure_api_key, azure_api_version, azure_deployment_name, threat_model_prompt)
                    elif model_provider == "OpenAI API":
                        model_output = get_threat_model(openai_api_key, selected_model, threat_model_prompt)
                    elif model_provider == "Google AI API":
                        model_output = get_threat_model_google(google_api_key, google_model, threat_model_prompt)
                    elif model_provider == "Mistral API":
                        model_output = get_threat_model_mistral(mistral_api_key, mistral_model, threat_model_prompt)
                    elif model_provider == "Ollama":
                        model_output = get_threat_model_ollama(ollama_model, threat_model_prompt)

                    # Access the threat model and improvement suggestions from the parsed content
                    threat_model = model_output.get("threat_model", [])
                    improvement_suggestions = model_output.get("improvement_suggestions", [])

                    # Save the threat model to the session state for later use in mitigations
                    st.session_state['threat_model'] = threat_model
                    break  # Exit the loop if successful
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        st.error(f"Error generating threat model after {max_retries} attempts: {e}")
                        threat_model = []
                        improvement_suggestions = []
                    else:
                        st.warning(f"Error generating threat model. Retrying attempt {retry_count+1}/{max_retries}...")

        # Convert the threat model JSON to Markdown
        markdown_output = json_to_markdown(threat_model, improvement_suggestions)

        # Display the threat model in Markdown
        st.markdown(markdown_output)

        # Add a button to allow the user to download the output as a Markdown file
        st.download_button(
            label="Download Threat Model",
            data=markdown_output,  # Use the Markdown output
            file_name="stride_gpt_threat_model.md",
            mime="text/markdown",
       )

# If the submit button is clicked and the user has not provided an application description
if threat_model_submit_button and not st.session_state.get('app_input'):
    st.error("Please enter your application details before submitting.")



# ------------------ Attack Tree Generation ------------------ #

with tab2:
    st.markdown("""
adalah cara terstruktur untuk menganalisis keamanan suatu sistem. Mereka mewakili skenario serangan potensial dalam format hierarkis, dengan tujuan akhir penyerang di akar dan berbagai jalur untuk mencapai tujuan tersebut sebagai cabang. Ini membantu dalam memahami kerentanan sistem dan memprioritaskan upaya mitigasi.
""")
    st.markdown("""---""")
    if model_provider == "Google AI API":
        st.warning("⚠️ Google's safety filters prevent the reliable generation of attack trees. Please use a different model provider.")
    elif model_provider == "Mistral API" and mistral_model == "mistral-small-latest":
            st.warning("⚠️ Mistral Small doesn't reliably generate syntactically correct Mermaid code. Please use the Mistral Large model for generating attack trees, or select a different model provider.")
    else:
        if model_provider == "Ollama":
            st.warning("⚠️ Users are likely to encounter syntax errors when generating attack trees using local LLMs. Experiment with different local LLMs to assess their output quality, or consider using a hosted model provider to generate attack trees.")
        
        # Create a submit button for Attack Tree
        attack_tree_submit_button = st.button(label="Generate Attack Tree")
        
        # If the Generate Attack Tree button is clicked and the user has provided an application description
        if attack_tree_submit_button and st.session_state.get('app_input'):
            app_input = st.session_state.get('app_input')
            # Generate the prompt using the create_attack_tree_prompt function
            attack_tree_prompt = create_attack_tree_prompt(app_type, authentication, internet_facing, sensitive_data, app_input)

            # Show a spinner while generating the attack tree
            with st.spinner("Generating attack tree..."):
                try:
                    # Call the relevant get_attack_tree function with the generated prompt
                    if model_provider == "Azure OpenAI Service":
                        mermaid_code = get_attack_tree_azure(azure_api_endpoint, azure_api_key, azure_api_version, azure_deployment_name, attack_tree_prompt)
                    elif model_provider == "OpenAI API":
                        mermaid_code = get_attack_tree(openai_api_key, selected_model, attack_tree_prompt)
                    elif model_provider == "Mistral API":
                        mermaid_code = get_attack_tree_mistral(mistral_api_key, mistral_model, attack_tree_prompt)
                    elif model_provider == "Ollama":
                        mermaid_code = get_attack_tree_ollama(ollama_model, attack_tree_prompt)

                    # Display the generated attack tree code
                    st.write("Attack Tree Code:")
                    st.code(mermaid_code)

                    # Visualise the attack tree using the Mermaid custom component
                    st.write("Attack Tree Diagram Preview:")
                    mermaid(mermaid_code)
                    
                    col1, col2, col3, col4, col5 = st.columns([1,1,1,1,1])
                    
                    with col1:              
                        # Add a button to allow the user to download the Mermaid code
                        st.download_button(
                            label="Download Diagram Code",
                            data=mermaid_code,
                            file_name="attack_tree.md",
                            mime="text/plain",
                            help="Download the Mermaid code for the attack tree diagram."
                        )

                    with col2:
                        # Add a button to allow the user to open the Mermaid Live editor
                        mermaid_live_button = st.link_button("Open Mermaid Live", "https://mermaid.live")
                    
                    with col3:
                        # Blank placeholder
                        st.write("")
                    
                    with col4:
                        # Blank placeholder
                        st.write("")
                    
                    with col5:
                        # Blank placeholder
                        st.write("")

                except Exception as e:
                    st.error(f"Error generating attack tree: {e}")


# ------------------ Mitigations Generation ------------------ #

with tab3:
    st.markdown("""
Gunakan tab ini untuk menghasilkan potensi mitigasi untuk ancaman yang diidentifikasi dalam model ancaman. Mitigasi adalah kontrol keamanan atau tindakan kontra yang dapat membantu mengurangi kemungkinan atau dampak dari ancaman keamanan. Mitigasi yang dihasilkan dapat digunakan untuk meningkatkan postur keamanan aplikasi dan melindungi dari serangan potensial.
""")
    st.markdown("""---""")
    
    # Create a submit button for Mitigations
    mitigations_submit_button = st.button(label="Suggest Mitigations")

    # If the Suggest Mitigations button is clicked and the user has identified threats
    if mitigations_submit_button:
        # Check if threat_model data exists
        if 'threat_model' in st.session_state and st.session_state['threat_model']:
            # Convert the threat_model data into a Markdown list
            threats_markdown = json_to_markdown(st.session_state['threat_model'], [])
            # Generate the prompt using the create_mitigations_prompt function
            mitigations_prompt = create_mitigations_prompt(threats_markdown)

            # Show a spinner while suggesting mitigations
            with st.spinner("Suggesting mitigations..."):
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        # Call the relevant get_mitigations function with the generated prompt
                        if model_provider == "Azure OpenAI Service":
                            mitigations_markdown = get_mitigations_azure(azure_api_endpoint, azure_api_key, azure_api_version, azure_deployment_name, mitigations_prompt)
                        elif model_provider == "OpenAI API":
                            mitigations_markdown = get_mitigations(openai_api_key, selected_model, mitigations_prompt)
                        elif model_provider == "Google AI API":
                            mitigations_markdown = get_mitigations_google(google_api_key, google_model, mitigations_prompt)
                        elif model_provider == "Mistral API":
                            mitigations_markdown = get_mitigations_mistral(mistral_api_key, mistral_model, mitigations_prompt)
                        elif model_provider == "Ollama":
                            mitigations_markdown = get_mitigations_ollama(ollama_model, mitigations_prompt)

                        # Display the suggested mitigations in Markdown
                        st.markdown(mitigations_markdown)
                        break  # Exit the loop if successful
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            st.error(f"Error suggesting mitigations after {max_retries} attempts: {e}")
                            mitigations_markdown = ""
                        else:
                            st.warning(f"Error suggesting mitigations. Retrying attempt {retry_count+1}/{max_retries}...")
            
            st.markdown("")

            # Add a button to allow the user to download the mitigations as a Markdown file
            st.download_button(
                label="Download Mitigations",
                data=mitigations_markdown,
                file_name="mitigations.md",
                mime="text/markdown",
            )
        else:
            st.error("Please generate a threat model first before suggesting mitigations.")

# ------------------ DREAD Risk Assessment Generation ------------------ #
with tab4:
    st.markdown("""
DREAD adalah metode untuk mengevaluasi dan memprioritaskan risiko yang terkait dengan ancaman keamanan. Ini menilai ancaman berdasarkan potensi Damage, Reproducibility, Exploitability, Affected users, dan Discoverability. Ini membantu dalam menentukan tingkat risiko keseluruhan dan fokus pada ancaman paling kritis terlebih dahulu. Gunakan tab ini untuk melakukan penilaian risiko DREAD untuk aplikasi/sistem Anda.
""")
    st.markdown("""---""")
    
    # Create a submit button for DREAD Risk Assessment
    dread_assessment_submit_button = st.button(label="Generate DREAD Risk Assessment")
    # If the Generate DREAD Risk Assessment button is clicked and the user has identified threats
    if dread_assessment_submit_button:
        # Check if threat_model data exists
        if 'threat_model' in st.session_state and st.session_state['threat_model']:
            # Convert the threat_model data into a Markdown list
            threats_markdown = json_to_markdown(st.session_state['threat_model'], [])
            # Generate the prompt using the create_dread_assessment_prompt function
            dread_assessment_prompt = create_dread_assessment_prompt(threats_markdown)
            # Show a spinner while generating DREAD Risk Assessment
            with st.spinner("Generating DREAD Risk Assessment..."):
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        # Call the relevant get_dread_assessment function with the generated prompt
                        if model_provider == "Azure OpenAI Service":
                            dread_assessment = get_dread_assessment_azure(azure_api_endpoint, azure_api_key, azure_api_version, azure_deployment_name, dread_assessment_prompt)
                        elif model_provider == "OpenAI API":
                            dread_assessment = get_dread_assessment(openai_api_key, selected_model, dread_assessment_prompt)
                        elif model_provider == "Google AI API":
                            dread_assessment = get_dread_assessment_google(google_api_key, google_model, dread_assessment_prompt)
                        elif model_provider == "Mistral API":
                            dread_assessment = get_dread_assessment_mistral(mistral_api_key, mistral_model, dread_assessment_prompt)
                        elif model_provider == "Ollama":
                            dread_assessment = get_dread_assessment_ollama(ollama_model, dread_assessment_prompt)
                        # Save the DREAD assessment to the session state for later use in test cases
                        st.session_state['dread_assessment'] = dread_assessment
                        break  # Exit the loop if successful
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            st.error(f"Error generating DREAD risk assessment after {max_retries} attempts: {e}")
                            dread_assessment = []
                        else:
                            st.warning(f"Error generating DREAD risk assessment. Retrying attempt {retry_count+1}/{max_retries}...")
            # Convert the DREAD assessment JSON to Markdown
            dread_assessment_markdown = dread_json_to_markdown(dread_assessment)
            # Display the DREAD assessment in Markdown
            st.markdown(dread_assessment_markdown)
            # Add a button to allow the user to download the test cases as a Markdown file
            st.download_button(
                label="Download DREAD Risk Assessment",
                data=dread_assessment_markdown,
                file_name="dread_assessment.md",
                mime="text/markdown",
            )
        else:
            st.error("Please generate a threat model first before requesting a DREAD risk assessment.")


# ------------------ Test Cases Generation ------------------ #

with tab5:
    st.markdown("""
digunakan untuk memvalidasi keamanan aplikasi dan memastikan bahwa kerentanan potensial diidentifikasi dan ditangani. Tab ini memungkinkan Anda untuk menghasilkan kasus uji menggunakan sintaks Gherkin. Gherkin menyediakan cara terstruktur untuk menjelaskan perilaku aplikasi dalam teks biasa, menggunakan sintaks sederhana dari pernyataan Given-When-Then. Ini membantu dalam membuat skenario pengujian yang jelas dan dapat dieksekusi.
""")
    st.markdown("""---""")
                
    # Create a submit button for Test Cases
    test_cases_submit_button = st.button(label="Generate Test Cases")

    # If the Generate Test Cases button is clicked and the user has identified threats
    if test_cases_submit_button:
        # Check if threat_model data exists
        if 'threat_model' in st.session_state and st.session_state['threat_model']:
            # Convert the threat_model data into a Markdown list
            threats_markdown = json_to_markdown(st.session_state['threat_model'], [])
            # Generate the prompt using the create_test_cases_prompt function
            test_cases_prompt = create_test_cases_prompt(threats_markdown)

            # Show a spinner while generating test cases
            with st.spinner("Generating test cases..."):
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        # Call to the relevant get_test_cases function with the generated prompt
                        if model_provider == "Azure OpenAI Service":
                            test_cases_markdown = get_test_cases_azure(azure_api_endpoint, azure_api_key, azure_api_version, azure_deployment_name, test_cases_prompt)
                        elif model_provider == "OpenAI API":
                            test_cases_markdown = get_test_cases(openai_api_key, selected_model, test_cases_prompt)
                        elif model_provider == "Google AI API":
                            test_cases_markdown = get_test_cases_google(google_api_key, google_model, test_cases_prompt)
                        elif model_provider == "Mistral API":
                            test_cases_markdown = get_test_cases_mistral(mistral_api_key, mistral_model, test_cases_prompt)
                        elif model_provider == "Ollama":
                            test_cases_markdown = get_test_cases_ollama(ollama_model, test_cases_prompt)

                        # Display the suggested mitigations in Markdown
                        st.markdown(test_cases_markdown)
                        break  # Exit the loop if successful
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            st.error(f"Error generating test cases after {max_retries} attempts: {e}")
                            test_cases_markdown = ""
                        else:
                            st.warning(f"Error generating test cases. Retrying attempt {retry_count+1}/{max_retries}...")
            
            st.markdown("")

            # Add a button to allow the user to download the test cases as a Markdown file
            st.download_button(
                label="Download Test Cases",
                data=test_cases_markdown,
                file_name="test_cases.md",
                mime="text/markdown",
            )
        else:
            st.error("Please generate a threat model first before requesting test cases.")