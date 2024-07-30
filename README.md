# Real-time Sentimental Analysis
<hr>
This repository contains code for a real-time emotion detection system using audio data. The system is built using machine learning techniques, including feature extraction, data augmentation, and a pre-trained neural network model.

## Features
<br>
Real-Time Detection: The system can analyze emotions in real-time from audio input.
Machine Learning Model: A pre-trained neural network model is used for accurate emotion prediction.
Data Augmentation: The training data is augmented to improve model robustness.

Model Loss - 0.464 when trained on 100 epochs

Predicted Emotions:
Anger(1), Disgust(2), Fear(3), Happy(4), Neutral(5), and Sad(6)
## Usage

1. **Preparation:**
   - Install the required dependencies using `pip install -r requirements.txt`.
   - Place your pre-trained model weights (`pretrained_model_weights.h5`) in the project directory.
   - Update the `model_architecture.json` file with the architecture of your pre-trained model.

2. **Run Real-Time Detection:**
   - Use the `real_time_detection` function in the provided Python script to perform real-time emotion detection on audio input.

3. **Customization:**
   - If you want to use your own pre-trained model, make sure to update the model architecture and weights accordingly.
   - Adjust feature extraction, data augmentation, or model parameters as needed.
<hr>
This project utilizes libraries such as Librosa, scikit-learn, and TensorFlow. Special thanks to the contributors of these open-source projects.

Feel free to explore, modify, and integrate this code into your projects. If you encounter issues or have suggestions, please open an issue.

import yaml
from typing import Dict, List, Any
import subprocess
import os

# ... (keep the existing parse_swagger and extract_api_details functions)

def generate_java_test_class(api_details: List[Dict[str, Any]], class_name: str, swagger_data: Dict[str, Any]) -> str:
    imports = generate_imports(api_details, swagger_data)
    
    java_code = f"""{imports}

@ExtendWith(PactConsumerTestExt.class)
@PactTestFor(providerName = "{class_name}Provider")
public class {class_name} {{

    private static final String PROVIDER_NAME = "{class_name}Provider";
    private static final String CONSUMER_NAME = "{class_name}Consumer";

    private String getAccessToken() {{
        return "accesstoken";  // Replace with actual token retrieval logic
    }}

    """

    for api in api_details:
        java_code += generate_pact_method(api)
        java_code += generate_test_method(api)

    java_code += "}\n"
    return java_code

def generate_imports(api_details: List[Dict[str, Any]], swagger_data: Dict[str, Any]) -> str:
    imports = set([
        "au.com.dius.pact.consumer.MockServer",
        "au.com.dius.pact.consumer.dsl.PactDslWithProvider",
        "au.com.dius.pact.consumer.junit5.PactConsumerTestExt",
        "au.com.dius.pact.consumer.junit5.PactTestFor",
        "au.com.dius.pact.core.model.RequestResponsePact",
        "au.com.dius.pact.core.model.annotations.Pact",
        "org.junit.jupiter.api.Test",
        "org.junit.jupiter.api.extension.ExtendWith",
        "org.springframework.http.HttpMethod",
        "org.springframework.http.ResponseEntity",
        "org.springframework.web.client.RestTemplate",
        "org.springframework.http.HttpHeaders",
        "org.springframework.http.HttpEntity",
        "static org.junit.jupiter.api.Assertions.assertEquals"
    ])

    # Add imports based on types used in the API
    for api in api_details:
        request_body = api.get('request_body', {})
        responses = api.get('responses', {})
        
        if request_body or responses:
            imports.add("com.fasterxml.jackson.databind.ObjectMapper")
        
        if any('file' in param.get('schema', {}).get('type', '').lower() for param in api.get('parameters', [])):
            imports.add("org.springframework.core.io.FileSystemResource")
            imports.add("org.springframework.util.LinkedMultiValueMap")
            imports.add("org.springframework.util.MultiValueMap")

    # Sort imports for consistency
    sorted_imports = sorted(imports)
    return "\n".join(f"import {imp};" for imp in sorted_imports)

# ... (keep the existing generate_pact_method, generate_test_method, and other helper functions)

def compile_java_code(class_name: str, java_file_path: str):
    # Ensure the necessary directories exist
    os.makedirs("target/classes", exist_ok=True)
    os.makedirs("target/test-classes", exist_ok=True)

    # Compile the Java code
    compile_command = [
        "javac",
        "-cp",
        "path/to/your/dependencies/*:target/classes",  # Replace with actual classpath
        "-d",
        "target/test-classes",
        java_file_path
    ]

    try:
        subprocess.run(compile_command, check=True)
        print(f"Successfully compiled {class_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling {class_name}: {e}")

def main(swagger_file: str):
    swagger_data = parse_swagger(swagger_file)
    api_details = extract_api_details(swagger_data)
    
    class_name = swagger_data.get('info', {}).get('title', 'API').replace(' ', '') + 'ContractTest'
    java_code = generate_java_test_class(api_details, class_name, swagger_data)
    
    # Write the generated Java code to a file
    java_file_path = f"{class_name}.java"
    with open(java_file_path, "w") as file:
        file.write(java_code)
    
    print(f"Generated Java test class: {java_file_path}")

    # Compile the generated Java code
    compile_java_code(class_name, java_file_path)

if __name__ == "__main__":
    main("path/to/your/swagger.yaml")
