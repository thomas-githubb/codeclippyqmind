from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from flask_cors import CORS
from llama_cpp import Llama

app = Flask(__name__)
CORS(app)


llm = Llama(
        model_path="ggml-model-f32.gguf",
        n_gpu_layers=-1,
        n_ctx=2048, 
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_code_completion/<string:user_input>')
def generate_code_completion(user_input):
    generated_code = generate_code(user_input)
    return jsonify({'generated_code': generated_code})

def generate_code(user_input):
    prompt = f"<s>[INST] Your task is to logically continue this code. Focus on generating a coherent and direct extension of the provided snippet. Consider the most likely context and functionality the initial code suggests, even if it's partially incomplete or ambiguous. Return only the Python code needed to complete or extend the snippet, without any additional comments or explanations. Your completion should be concise yet functional, aiming for effectiveness rather than reaching the maximum token length of 1000. In cases where the input might imply multiple directions, opt for the most common or practical solution. Wrap any code generated in ``` and ```. This is the given code prompt: {user_input} [/INST]"

    output = llm(
        prompt=prompt,
        max_tokens=None, # Generate up to 32 tokens, set to None to generate up to the end of the context window
        echo=False # Echo the prompt back in the output
    ) 

    generated_code = output["choices"][0]["text"]
    print(output)

    # Extract code inside triple quotes if they exist
    start_index = generated_code.find("```")
    if start_index == -1:
        start_index = generated_code.find("```python")
    end_index = generated_code.rfind("```")

    if start_index != -1 and end_index != -1:
        generated_code = generated_code[start_index + 3:end_index].strip()

    return generated_code

if __name__ == '__main__':
    app.run(debug=False)
