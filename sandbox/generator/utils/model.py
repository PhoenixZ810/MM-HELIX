import os
import sys
sys.path.append("/Users/linjunming/Desktop/工作/MMRL/VLMEvalKit")
# from vlmeval.api import OpenAIWrapper
import time
os.environ['OPENAI_API_KEY'] = 'sk-SIIoxy6kU008sVj7Ef63995c0b7b47E092C9904324236a2a'
os.environ['OPENAI_API_BASE'] = 'http://152.69.226.145:3000/v1/chat/completions'
import openai, os, json, tqdm
import glob
import base64
os.environ['OPENAI_API_KEY'] = 'sk-8ltww2EEC4ARZHxOTKBLUgSsscAK4LekvihLYuwt9yDb0UJ3'  # Replace with your actual key
os.environ['OPENAI_BASE_URL'] = 'https://api.deepbricks.ai/v1/'  # Replace if necessary
model_openai = openai.OpenAI()
from openai import OpenAI
def encode_image(image_path):
    """将图像文件编码为 base64 字符串。"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def gemini(model_input, image_path):
    from google import genai
    client = genai.Client(api_key="AIzaSyBAQdnRPbJ3frh0yx7ZGNDVs4RCKhIZLg0")

    myfile = client.files.upload(file=image_path)

    response = client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25",
        contents=[myfile, model_input])
    print(response.text)
    return response.text


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def R1(model_input, image_path=None):
    client = OpenAI(api_key='YOUR_API_KEY', base_url="http://152.69.226.145:57932/v1")
    model_name = client.models.list().data[0].id
    response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": model_input},
                    ],
                    temperature=0.001,
                    timeout=100,
                    max_tokens=16384
                )
    return response.choices[0].message.content

# def R1(model_input, image_path=None):
#     response = client.chat.completions.create(
#     model="deepseek-reasoner",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": model_input},
#     ],
#     stream=False
#     )
#     print(response.choices[0].message.content)
#     return response.choices[0].message.content     
def gpt_4o(model_input, image_path):
    model_input = [{'type': 'text', 'text': model_input}]
    if image_path:
        base64_image = encode_image(image_path)
        model_input.append({
            'type': 'image_url',
            'image_url': {'url': f"data:image/jpeg;base64,{base64_image}", 'detail': 'low'},
        })
    else:
        print(f"Warning: No image path provided for question: {model_input}")
    response = ''
    for _ in range(20):
        try:
            response = model_openai.chat.completions.create(
                model='gpt-4o',  # Or your preferred model
                messages=[{
                    'role': 'user',
                    'content': model_input
                }],
                max_tokens=1000,
            ).choices[0].message.content
            # print("fail") # Removed the print statement, as it always prints "fail" even on success.
            break  # Exit the loop on success

        except Exception as e:  # Catch specific exceptions if possible
            print(f"An error occurred: {e}") # Print the error for debugging
            continue

    return response

# def gpt_4o(model_input,image_path=None):
#     for _ in range(20):
#         try:
#             response = model_openai.chat.completions.create(
#                 model='gpt-4o',  # Or your preferred model
#                 messages=[{
#                     'role': 'user',
#                     'content': model_input
#                 }],
#                 max_tokens=1000,
#             ).choices[0].message.content
#             # print("fail") # Removed the print statement, as it always prints "fail" even on success.
#             break  # Exit the loop on success

#         except Exception as e:  # Catch specific exceptions if possible
#             print(f"An error occurred: {e}") # Print the error for debugging
#             continue

#     return response


# def gpt_4o(model_input,image_path=None):
#     gpt4o = OpenAIWrapper(
#     'gpt-4o-2024-11-20',
#     temperature=0,
#     max_tokens=4096,
#     img_detail='high',
#     # img_size=768,
#     timeout=300,
# )
#     message = []
#     text = {'type': 'text', 'value': model_input}
#     message.append(text)
#     if image_path != None:
#         image = {'type': 'image', 'value': image_path}
#         message.append(image)
#     ans = gpt4o.generate_inner(message)
#     return ans[1]

# def gemini(model_input,image_path=None):
#     gemini2d0pro = OpenAIWrapper(
#     'gemini-1.5-pro',
#     temperature=0,
#     max_tokens=4096,
#     img_detail='high',
#     # img_size=768,
#     timeout=300,
# )
#     message = []
#     text = {'type': 'text', 'value': model_input}
#     message.append(text)
#     if image_path != None:
#         image = {'type': 'image', 'value': image_path}
#         message.append(image)
#     ans = gemini2d0pro.generate_inner(message)
#     return ans[1]

# def gemini(model_input=None, image_path=None):
#     import time    
#     client = genai.Client(api_key="AIzaSyBAQdnRPbJ3frh0yx7ZGNDVs4RCKhIZLg0")
    
#     max_retries = 20
#     retry_count = 0
    
#     while retry_count < max_retries:
#         try:
#             response = client.models.generate_content(
#                 model="gemini-2.5-pro-exp-03-25", contents=model_input
#             )
#             return response.text
#         except Exception as e:
#             retry_count += 1
#             print(f"Attempt {retry_count} failed with error: {e}")
            
#             if retry_count >= max_retries:
#                 print(f"Failed after {max_retries} attempts")
#                 raise
            
def claude(model_input,image_path=None):
    claude = OpenAIWrapper(
    'claude-3-7-sonnet-20250219',
    temperature=0,
    max_tokens=4096,
    img_detail='high',
    # img_size=768,
    timeout=300,
)
    message = []
    text = {'type': 'text', 'value': model_input}
    message.append(text)
    if image_path != None:
        image = {'type': 'image', 'value': image_path}
        message.append(image)
    
    ans = claude.generate_inner(message)
    return ans[1]







