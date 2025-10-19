import sys
import time
os.environ['OPENAI_API_KEY'] = 
os.environ['OPENAI_API_BASE'] = 
import openai, os, json, tqdm
import glob
import base64


model_openai = openai.OpenAI()
from openai import OpenAI


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def gemini(model_input, image_path):
    from google import genai
    client = genai.Client(api_key="")

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
    client = OpenAI(api_key='YOUR_API_KEY', base_url="")
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
            break

        except Exception as e:  # Catch specific exceptions if possible
            print(f"An error occurred: {e}") # Print the error for debugging
            continue

    return response

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







