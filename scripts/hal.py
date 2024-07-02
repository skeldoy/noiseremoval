# This is a self-replicating python program that will use LLM to generate a python program that uses LLM to generate a python program etc etc 

import ollama
import subprocess
import logging
import tempfile
import os
import re
import sys


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_cleaner(input_text: str) -> str:
    try:
        result = subprocess.run(['perl', 'cleaner.pl'], input=input_text, capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        logging.error("* Error running cleaner.pl: %s", str(e))
        return input_text

def clean_response(response: str) -> str:
    # Remove anything that's not Python code using regex
    code_block_pattern = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)
    match = code_block_pattern.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()

def clean_response_space(code: str) -> str:
    lines = code.split('\n')
    cleaned_lines = [line.lstrip() for line in lines]
    return '\n'.join(cleaned_lines)

def format_code_with_black(code: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as tmp_file:
        tmp_file.write(code.encode())
        tmp_file_path = tmp_file.name

    try:
        result = subprocess.run(['black', tmp_file_path], capture_output=True, text=True)
        if result.returncode != 0:
            logging.error("* Error formatting code with black: %s", result.stderr)
            return code

        with open(tmp_file_path, 'r') as tmp_file:
            formatted_code = tmp_file.read()

        return formatted_code
    except Exception as e:
        logging.error("* Error running black: %s", str(e))
        return code
    finally:
        os.remove(tmp_file_path)

def run_code(code: str) -> str:
    try:
        with open('temp_code.py', 'w') as f:
            f.write(code)
        result = subprocess.run(['python', 'temp_code.py'], capture_output=True, text=True)
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)

def evaluate_output(output: str) -> bool:
    if "Error" in output or "Exception" in output or "IndentationError" in output or "SyntaxError" in output:
        return False
    return True

def generate_initial_code(prompt: str, history: list) -> str:
    user_message = {
        'role': 'user',
        'content': prompt
    }
    history.append(user_message)
    
    logging.info("* Generating initial code...")
    response = ollama.chat(model='mycoder', messages=history)
    code = response['message']['content']
    
    assistant_message = {
        'role': 'assistant',
        'content': code
    }
    history.append(assistant_message)
    
    logging.info("* Initial code generated:\n%s", code)
    code = clean_response(code)
    #code = clean_response_space(code)
    code = format_code_with_black(code)
    return code

def improve_code(code: str, feedback: str, history: list) -> str:
    user_message = {
        'role': 'user',
        'content': f"Improve the following code based on this feedback: {feedback}\n\nCode:\n\n{code}\n\nKeep in mind: Respond with only pure Python code. No comments, no explanations, no meta data. Ensure proper indentation and start the response without leading spaces or tabs. Do not include ```python or any triple backticks. Make sure that you think about this as meta-programming and not programming."
    }
    history.append(user_message)
    
    logging.info("* Requesting code improvement...")
    response = ollama.chat(model='mothercoder', messages=history)
    improved_code = response['message']['content']
    
    assistant_message = {
        'role': 'assistant',
        'content': improved_code
    }
    history.append(assistant_message)
    
    logging.info("* Improved code received:\n%s", improved_code)
    improved_code = clean_response(improved_code)
    #improved_code = clean_response_space(improved_code)
    improved_code = format_code_with_black(improved_code)
    return improved_code

# Initialize conversation history
conversation_history = []

firstpurpose = """ Generate a python program that generates a python program using ollama to query an LLM about writing meta-programming python code to generate python code using ollama\n\n example code:\n\n import ollama \n  response = ollama.chat(model="mycoder", messages=[{"role": "user","content": "generate python code that uses ollama to generate python code that generates python code. The python code should be instructed to generate python code that is executable, then execute it. The way to do it is: import ollama \n  response = ollama.chat(model="mycoder", messages=[{"role": "user","content": "generate python code that uses ollama to generate python code that generates python code. The python code should be instructed to generate python code that is executable, then execute it. The execution will provide another python program that will use ollama to generate another python program. " }]) \n print(response["message"]["content"]). The execution will provide another python program that will use ollama to generate another python program. " }]) \n print(response["message"]["content"]). Save the code to disk and execute it, then evaluate the response from the execution by using another query to ollama. Repeat this step 5 times. Make sure that the next python script understands that this should go on until it breaks. By adding a lot of this initial prompt to the prompts of the future queries we increase the chance of recursion. Remember to instruct the python program to make sure the generated program logs or prints to screen about what it is doing. """    

purpose = "%s this is the code that generated this. Improve it further: %s" % (firstpurpose, firstpurpose)

# Read in our own source code for adding to the context:
with open(sys.argv[0], 'r') as file:
    source_code = file.read()
this_source = "This is the source code for the mother process <code> %s </code>. "

initial_prompt = (
    f"Generate a python application that {purpose}. No comments or explanation. "
    f"Just the actual code for {purpose}, please. Be careful about indentation! "
    f"You don't have to explicitly say or mention that it is python code. "
    f"We just want pure python code without any meta data around it. Be very careful with the indenting. "
    f"You must never start the response with an indent like tab or spaces. Never output ```python or any triple backticks. "
    f"For every iteration of code generation we will try to run the code you generate. You will have to carry as much of this prompt forward to enable future generations of the code to be able to instruct the LLM properly. "
    f"The output of the execution will be added as a feedback to the LLM for evaluation. Be very verbose in the prompting of the LLM. "
    f"The output will grow after every unsuccessful iteration and become a long list of things that went wrong. "
    f"Try to rectify the situation by carefully analyzing the context to determine what went wrong in the code generation. "
    f"Most likely there is a lot of things in the output of the LLM that is not pure python; explanations, comments, wrong indenting... "
    f"Try to figure this out and provide the correct python code without explaining or apologizing about it. If you are building queries"
    f"for future ollama queries for generating pyton code please include instrucions like you have been given to the LLMs so they don't start generating comments or other irrelevant output."
    f"Try to make the point to the future python scripts you generate that this is a recursion: The more of this instruction is passed on the better it is going to work."
)


initial_code = generate_initial_code(initial_prompt, conversation_history)
#initial_code = generate_initial_code(this_source, conversation_history)
print(f"\n *** This is the full context: %s", initial_code, " \n **** \n")

output = run_code(initial_code)
logging.info("* Initial code output:\n%s", output)
is_correct = evaluate_output(output)

max_iterations = 5
iterations = 0

while not is_correct and iterations < max_iterations:
    feedback = f"The code did not run correctly. Output was: {output}"
    logging.info("* Feedback for improvement: %s", feedback)
    initial_code = improve_code(initial_code, feedback, conversation_history)
    output = run_code(initial_code)
    logging.info("* Improved code output:\n%s", output)
    is_correct = evaluate_output(output)
    iterations += 1

if is_correct:
    logging.info("* Code runs flawlessly.")
else:
    logging.info("* Failed to generate correct code after several iterations.")
