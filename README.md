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

class TestMethodGenerator(CodeGenerator):
    def generate(self, api: Dict[str, Any]) -> str:
        method_name = f"test_{api['operation_id']}"
        return f"""
    @Test
    @PactTestFor(pactMethod = "{api['operation_id']}")
    void {method_name}(MockServer mockServer) {{
        {self.generate_http_client(api)}
    }}
    """

    def generate_http_client(self, api: Dict[str, Any]) -> str:
        return f"""
        Response response = given()
            .baseUri(mockServer.getUrl())
            {self.generate_headers(api)}
            {self.generate_query_params(api)}
            {self.generate_request_body(api)}
        .when()
            .{api['method'].lower()}("{api['path']}")
        .then()
            .statusCode(200)
            {self.generate_response_assertions(api)}
            .extract().response();

        // Additional assertions can be added here if needed
        """

    def generate_headers(self, api: Dict[str, Any]) -> str:
        headers = [param for param in api['parameters'] if param['in'] == 'header']
        if not headers:
            return '.header("Content-Type", "application/json")'
        header_str = ""
        for header in headers:
            header_str += f'.header("{header["name"]}", "{header["schema"]["type"]}")\n            '
        return header_str

    def generate_query_params(self, api: Dict[str, Any]) -> str:
        query_params = [param for param in api['parameters'] if param['in'] == 'query']
        if not query_params:
            return ""
        param_str = ""
        for param in query_params:
            param_str += f'.queryParam("{param["name"]}", "{param["schema"].get("example", "example")}")\n            '
        return param_str

    def generate_request_body(self, api: Dict[str, Any]) -> str:
        if 'requestBody' not in api or 'content' not in api['requestBody']:
            return ""
        content = api['requestBody']['content']
        if 'application/json' in content:
            return '.body(generateRequestBody())'
        return ""

    def generate_response_assertions(self, api: Dict[str, Any]) -> str:
        if '200' not in api['responses']:
            return ""
        response = api['responses']['200']
        if 'content' not in response or 'application/json' not in response['content']:
            return ""
        schema = response['content']['application/json']['schema']
        return '.body(matchesJsonSchema(generateResponseSchema()))'

        class JavaTestClassBuilder:
    def __init__(self, class_name: str, swagger_data: Dict[str, Any]):
        self.class_name = class_name
        self.swagger_data = swagger_data
        self.imports = set()
        self.methods = []

    def generate_imports(self):
        imports = [
            "import au.com.dius.pact.consumer.MockServer;",
            "import au.com.dius.pact.consumer.dsl.PactDslWithProvider;",
            "import au.com.dius.pact.consumer.junit5.PactConsumerTestExt;",
            "import au.com.dius.pact.consumer.junit5.PactTestFor;",
            "import au.com.dius.pact.core.model.RequestResponsePact;",
            "import au.com.dius.pact.core.model.annotations.Pact;",
            "import org.junit.jupiter.api.Test;",
            "import org.junit.jupiter.api.extension.ExtendWith;",
            "import io.restassured.response.Response;",
            "import static io.restassured.RestAssured.given;",
            "import static io.restassured.module.jsv.JsonSchemaValidator.matchesJsonSchema;",
            "import org.json.JSONObject;",
            "import org.json.JSONArray;"
        ]
        return "\n".join(imports)

    def build(self) -> str:
        imports = self.generate_imports()
        methods = "\n\n".join(self.methods)
        
        return f"""{imports}

@ExtendWith(PactConsumerTestExt.class)
@PactTestFor(providerName = "{self.class_name}Provider")
public class {self.class_name} {{

    private static final String PROVIDER_NAME = "{self.class_name}Provider";
    private static final String CONSUMER_NAME = "{self.class_name}Consumer";

    private String getAccessToken() {{
        return "accesstoken";  // Replace with actual token retrieval logic
    }}

    private JSONObject generateRequestBody() {{
        // Implement request body generation based on Swagger spec
        return new JSONObject();
    }}

    private String generateResponseSchema() {{
        // Implement response schema generation based on Swagger spec
        return "{{}}";
    }}

    {methods}
}}
"""


class PactMethodGenerator(CodeGenerator):
    def generate(self, api: Dict[str, Any]) -> str:
        method_name = f"pact_{api['operation_id']}"
        return f"""
    @Pact(provider = PROVIDER_NAME, consumer = CONSUMER_NAME)
    public RequestResponsePact {method_name}(PactDslWithProvider builder) {{
        return builder
            .given("{api['summary']}")
            .uponReceiving("{api['description']}")
            .path("{api['path']}")
            .method("{api['method']}")
            {self.generate_headers(api)}
            {self.generate_query_parameters(api)}
            {self.generate_request_body(api)}
            .willRespondWith()
            .status(200)
            {self.generate_response(api)}
            .toPact();
    }}
    """

    def generate_headers(self, api: Dict[str, Any]) -> str:
        headers = [param for param in api['parameters'] if param['in'] == 'header']
        if not headers:
            return ""
        header_str = ".headers()"
        for header in headers:
            header_str += f"\n            .header(\"{header['name']}\", \"{header['schema']['type']}\")"
        return header_str

    def generate_query_parameters(self, api: Dict[str, Any]) -> str:
        query_params = [param for param in api['parameters'] if param['in'] == 'query']
        if not query_params:
            return ""
        param_str = ".query()"
        for param in query_params:
            param_str += f"\n            .parameter(\"{param['name']}\", \"{param['schema']['type']}\")"
        return param_str

    def generate_request_body(self, api: Dict[str, Any]) -> str:
        if 'requestBody' not in api or 'content' not in api['requestBody']:
            return ""
        content = api['requestBody']['content']
        if 'application/json' in content:
            schema = content['application/json']['schema']
            return f".body(PactDslJsonBody.{self.generate_json_body(schema)})"
        return ""

    def generate_json_body(self, schema: Dict[str, Any]) -> str:
        if schema['type'] == 'object':
            body = "newJsonBody()"
            for prop, prop_schema in schema['properties'].items():
                body += f".{prop_schema['type']}(\"{prop}\", \"{prop_schema.get('example', 'example')}\")"
            return body + ".build()"
        elif schema['type'] == 'array':
            return f"newJsonArray().{schema['items']['type']}(\"example\").build()"
        else:
            return f"{schema['type']}()"

    def generate_response(self, api: Dict[str, Any]) -> str:
        if '200' not in api['responses']:
            return ""
        response = api['responses']['200']
        if 'content' not in response or 'application/json' not in response['content']:
            return ""
        schema = response['content']['application/json']['schema']
        return f".body(PactDslJsonBody.{self.generate_json_body(schema)})"

        
import yaml
from abc import ABC, abstractmethod
from typing import Dict, List, Any
import subprocess
import os

class SwaggerParser:
    @staticmethod
    def parse(file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

class ApiExtractor:
    @staticmethod
    def extract_details(swagger_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        api_details = []
        for path, path_item in swagger_data['paths'].items():
            for method, operation in path_item.items():
                api_details.append({
                    'path': path,
                    'method': method.upper(),
                    'operation_id': operation.get('operationId', f"{method}_{path.replace('/', '_')}"),
                    'summary': operation.get('summary', ''),
                    'description': operation.get('description', ''),
                    'parameters': operation.get('parameters', []),
                    'responses': operation.get('responses', {}),
                    'request_body': operation.get('requestBody', {})
                })
        return api_details

class CodeGenerator(ABC):
    @abstractmethod
    def generate(self, api: Dict[str, Any]) -> str:
        pass

class PactMethodGenerator(CodeGenerator):
    def generate(self, api: Dict[str, Any]) -> str:
        method_name = f"pact_{api['operation_id']}"
        return f"""
    @Pact(provider = PROVIDER_NAME, consumer = CONSUMER_NAME)
    public RequestResponsePact {method_name}(PactDslWithProvider builder) {{
        return builder
            .given("{api['summary']}")
            .uponReceiving("{api['description']}")
            .path("{api['path']}")
            .method("{api['method']}")
            .willRespondWith()
            .status(200)
            .toPact();
    }}
    """

class TestMethodGenerator(CodeGenerator):
    def generate(self, api: Dict[str, Any]) -> str:
        method_name = f"test_{api['operation_id']}"
        return f"""
    @Test
    @PactTestFor(pactMethod = "{api['operation_id']}")
    void {method_name}(MockServer mockServer) {{
        // Implement test logic here
    }}
    """

class HeadersGenerator(CodeGenerator):
    def generate(self, api: Dict[str, Any]) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.get_access_token()
        }
        return ", ".join([f'"{k}", "{v}"' for k, v in headers.items()])

class RequestBodyGenerator(CodeGenerator):
    def generate(self, api: Dict[str, Any]) -> str:
        # This is a simplified implementation. You might want to generate
        # a more complex request body based on the API specification.
        return '"{}"'

class CodeGeneratorFactory:
    @staticmethod
    def create_generator(generator_type: str) -> CodeGenerator:
        if generator_type == 'pact_method':
            return PactMethodGenerator()
        elif generator_type == 'test_method':
            return TestMethodGenerator()
        elif generator_type == 'headers':
            return HeadersGenerator()
        elif generator_type == 'request_body':
            return RequestBodyGenerator()
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")

class JavaTestClassBuilder:
    def __init__(self, class_name: str, swagger_data: Dict[str, Any]):
        self.class_name = class_name
        self.swagger_data = swagger_data
        self.imports = set()
        self.methods = []

    def add_import(self, import_statement: str):
        self.imports.add(import_statement)
        return self

    def add_method(self, method: str):
        self.methods.append(method)
        return self

    def generate_imports(self):
        imports = [
            "import au.com.dius.pact.consumer.MockServer;",
            "import au.com.dius.pact.consumer.dsl.PactDslWithProvider;",
            "import au.com.dius.pact.consumer.junit5.PactConsumerTestExt;",
            "import au.com.dius.pact.consumer.junit5.PactTestFor;",
            "import au.com.dius.pact.core.model.RequestResponsePact;",
            "import au.com.dius.pact.core.model.annotations.Pact;",
            "import org.junit.jupiter.api.Test;",
            "import org.junit.jupiter.api.extension.ExtendWith;"
        ]
        return "\n".join(imports)

    def build(self) -> str:
        imports = self.generate_imports()
        methods = "\n\n".join(self.methods)
        
        return f"""{imports}

@ExtendWith(PactConsumerTestExt.class)
@PactTestFor(providerName = "{self.class_name}Provider")
public class {self.class_name} {{

    private static final String PROVIDER_NAME = "{self.class_name}Provider";
    private static final String CONSUMER_NAME = "{self.class_name}Consumer";

    private String getAccessToken() {{
        return "accesstoken";  // Replace with actual token retrieval logic
    }}

    {methods}
}}
"""

class JavaCompiler:
    @staticmethod
    def compile(class_name: str, java_file_path: str):
        os.makedirs("target/classes", exist_ok=True)
        os.makedirs("target/test-classes", exist_ok=True)

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

class ContractTestGenerator:
    def __init__(self, swagger_file: str):
        self.swagger_file = swagger_file
        self.swagger_data = SwaggerParser.parse(swagger_file)
        self.api_details = ApiExtractor.extract_details(self.swagger_data)
        info = self.swagger_data.get('info', {})
        self.class_name = info.get('title', 'API').replace(' ', '') + 'ContractTest'

    def generate(self):
        builder = JavaTestClassBuilder(self.class_name, self.swagger_data)
        
        pact_generator = CodeGeneratorFactory.create_generator('pact_method')
        test_generator = CodeGeneratorFactory.create_generator('test_method')

        for api in self.api_details:
            builder.add_method(pact_generator.generate(api))
            builder.add_method(test_generator.generate(api))

        java_code = builder.build()
        
        java_file_path = f"{self.class_name}.java"
        with open(java_file_path, "w") as file:
            file.write(java_code)
        
        print(f"Generated Java test class: {java_file_path}")
        
        JavaCompiler.compile(self.class_name, java_file_path)

def main(swagger_file: str):
    generator = ContractTestGenerator(swagger_file)
    generator.generate()

if __name__ == "__main__":
    main("path/to/your/swagger.yaml")
