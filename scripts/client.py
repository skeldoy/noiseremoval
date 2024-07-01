from ollama import Client
import os 
import subprocess
#from subprocess import Popen, PIPE, CalledProcessError
import inspect
import sys
import time
from datetime import datetime


my_source = (inspect.getsource(sys.modules[__name__]))


# Initialize clients for both machines
first_client = Client(host="http://vanadium.local:10000")
second_client = Client(host="http://titan-pc.local:10000")

# Start the conversation with a greeting
message_content = """ 
#Lets write a python program that talks to an LLM:

from ollama import Client
import subprocess
import os
import sys
import inspect
import time
from datetime import datetime
first_client = Client(host="http://vanadium.local:10000")
second_client = Client(host="http://titan-pc.local:10000")
message_content = "Write a python program that talks to an LLM about writing a python program that talks to an LLM. Save the generated code to file before executing it. Use python3"
my_source = (inspect.getsource(sys.modules[__name__]))
while True:
    # Send message to first machine
    output = ""
    response = first_client.chat(model="llama3", messages=[{"role": "user", "content": message_content}])
    answer = response["message"]["content"]
    print("First Machine:")
    print(answer)
    print("-----------------------------------------------------------------------")
    if '```' in answer and '```' in answer:  # assuming the LLM wraps code with backticks
        start = answer.find('```') + len('```')
        end = answer.rfind('```')
        code_to_execute = answer[start:end]
    else:
        print("First Machine didn't provide any Python code.")
        break
    try:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f'btemp_{timestamp}.py'

        with open(filename, 'w') as file:
            file.write(code_to_execute)
        cmd = "/usr/bin/python3"
        print("----------------------------------------------------------------------- Executing:")
        #subprocess.check_output([cmd, 'temp.py'], shell=True, universal_newlines=True, stderr=subprocess.STDOUT)
        process = subprocess.Popen(['python3', filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        #output = subprocess.check_output(["python3", "-c", f"{code_to_execute}"], stderr=subprocess.STDOUT, universal_newlines=True, stdout=subprocess.PIPE,bufsize=1)
        print(f"Executing code: {code_to_execute}")
        print("Output:")
        print(output.strip())
    except Exception as e:
        print(f"Error executing code: {e}")
    print("----------------------------------------------------------------------- Execution done..")
    # Send the answer to second machine and get its response
    response = second_client.chat(model="llama3", messages=[{"role": "user", "content": output}])
    second_answer = response["message"]["content"]
    print("Second Machine:")
    print(second_answer)
    print("-----------------------------------------------------------------------")

    # Prepare for the next round of conversation
    message_content = second_answer


#Now give it the capability to execute code on the first machine.
#And then evaluate the output of the execution on the second machine.
#The goal is to build a system that makes two LLMs talk to each other to create and run code.
#This primary run saves files as atemp and instructs the next LLM to save them as btemp .. and the next instruction should be ctemp.. etc etc
"""
print(message_content)

while True:
    # Send message to first machine
    output = ""
    response = first_client.chat(model="llama3", messages=[{"role": "user", "content": message_content}])
    answer = response["message"]["content"]
    print("First Machine:")
    print(answer)
    print("-----------------------------------------------------------------------")
    if '```python' in answer and '```' in answer or '```' in answer:  # assuming the LLM wraps code with backticks
        start = answer.find('```') + len('```')
        end = answer.rfind('```')
        if 'python' in answer[start:end]: #remove python
            start += len('python')
        code_to_execute = answer[start:end]
    else:
        print("First Machine didn't provide any Python code.")
        break
    try:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f'atemp_{timestamp}.py'

        with open(filename, 'w') as file:
            file.write(code_to_execute)
        cmd = "/usr/bin/python3"
        print("----------------------------------------------------------------------- Executing:")
        #subprocess.check_output([cmd, 'temp.py'], shell=True, universal_newlines=True, stderr=subprocess.STDOUT)
        process = subprocess.Popen(['python3', filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        #output = subprocess.check_output(["python3", "-c", f"{code_to_execute}"], stderr=subprocess.STDOUT, universal_newlines=True, stdout=subprocess.PIPE,bufsize=1)
        print(f"Executing code: {code_to_execute}")
        print("Output:")
        print(output.strip())
    except Exception as e:
        print(f"Error executing code: {e}")
    print("----------------------------------------------------------------------- Execution done..")
    # Send the answer to second machine and get its response
    next_instruction = "The last LLM generated: [\n''' ", f'{code_to_execute}', "  ''' \n ] and the output of running it was: ''' ", f'{output}', " ''' "
    response = second_client.chat(model="llama3", messages=[{"role": "user", "content": next_instruction}])
    second_answer = response["message"]["content"]
    print("Second Machine:")
    print(second_answer)
    print("-----------------------------------------------------------------------")

    # Prepare for the next round of conversation
    message_content = second_answer


