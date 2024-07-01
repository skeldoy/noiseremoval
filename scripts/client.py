from ollama import Client

import inspect
import sys

my_source = (inspect.getsource(sys.modules[__name__]))


# Initialize clients for both machines
first_client = Client(host="http://vanadium.local:10000")
second_client = Client(host="http://titan-pc.local:10000")

# Start the conversation with a greeting
message_content = """ Lets write a python program that talks to an LLM 

Use this \n \n \n  %s  \n \n \n as a starting point 

to make a python program and execute the python program. """, my_source

print(message_content)

while True:
    # Send message to first machine
    response = first_client.chat(model="llama3", messages=[{"role": "user", "content": message_content}])
    answer = response["message"]["content"]
    print("First Machine:")
    print(answer)
    print("-----------------------------------------------------------------------")

    # Send the answer to second machine and get its response
    response = second_client.chat(model="llama3", messages=[{"role": "user", "content": answer}])
    second_answer = response["message"]["content"]
    print("Second Machine:")
    print(second_answer)
    print("-----------------------------------------------------------------------")

    # Prepare for the next round of conversation
    message_content = second_answer


