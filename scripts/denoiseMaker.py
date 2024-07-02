from ollama import Client
import subprocess
import os
import sys
import inspect
import time
from datetime import datetime

# Get the source code of the current module
my_source = inspect.getsource(sys.modules[__name__])

# Initialize clients for both machines
first_client = Client(host="http://localhost:11434")
second_client = Client(host="http://beast.skeldoy.no:11434")

# Start the conversation with a greeting and initial instructions
initial_message = """

Write a python program that uses pytorch to create a neural network to train a model to denoise images.
The images are in ../data/*.png and are of various dimensions. We don't have any clean images, just
noisy images. So the type of neural network is up to you. Previous experiments have favoured cycleGAN-networks.
The model should be able to denoise the images.
After the model has trained for a while it has to be written to disk. 
Then the model should be loaded and used to denoise the images in../data/*.png
The output of the denoised image is saved as../data/denoised/*.png.

"""

print(initial_message)

# Function to extract and execute code from LLM response
def extract_and_execute_code(response, temp_prefix):
    if '```python' in response and '```' in response:
        start = response.find('```python') + len('```python')
        end = response.rfind('```')
    elif '```' in response:
        start = response.find('```') + len('```')
        end = response.rfind('```')
    elif '```Python' in response:
        start = response.find('```Python') + len('```')
        end = response.rfind('```')
    else:
        print("No code found in the response.")
        return None, None

    code_to_execute = response[start:end].strip()
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f'{temp_prefix}_{timestamp}.py'

    with open(filename, 'w') as file:
        file.write(code_to_execute)
    
    try:
        process = subprocess.Popen(['python3', filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = ""
        while True:
            line = process.stdout.readline()
            if line == '' and process.poll() is not None:
                break
            if line:
                output += line
                print(line.strip())
        return code_to_execute, output.strip()
    except Exception as e:
        print(f"Error executing code: {e}")
        return code_to_execute, str(e)

# Main loop for conversation between LLMs
message_content = initial_message
while True:
    # Send message to the first machine
    response = first_client.chat(model="codestral", messages=[{"role": "user", "content": message_content}])
    answer = response["message"]["content"]
    print("First Machine:")
    print(answer)
    print("-----------------------------------------------------------------------")

    code_to_execute, execution_output = extract_and_execute_code(answer, "atemp")
    if code_to_execute is None:
        break

    print("Executing code:")
    print(code_to_execute)
    print("Output:")
    print(execution_output)
    print("-----------------------------------------------------------------------")

    # Prepare the next instruction for the second machine
    next_instruction = (
        f"The last LLM generated the following code:\n```python\n{code_to_execute}\n```\n"
        f"And the output of running it was:\n```\n{execution_output}\n```"
    )

    # Send the result to the second machine and get its response
    response = second_client.chat(model="mycoder", messages=[{"role": "user", "content": next_instruction}])
    second_answer = response["message"]["content"]
    print("Second Machine:")
    print(second_answer)
    print("-----------------------------------------------------------------------")

    code_to_execute, execution_output = extract_and_execute_code(second_answer, "btemp")
    if code_to_execute is None:
        break
    print("Executing code:")
    print(code_to_execute)
    print("Output:")
    print(execution_output)
    print("________________________________________________________________________")


    # Update the message content for the next iteration
    message_content = ( f"The last LLM generated the following response:\n```python\n{second_answer}\n```\n"
                        f"And the output of running it was:\n```\n{execution_output}\n```")

