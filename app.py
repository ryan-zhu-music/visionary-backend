from flask import Flask, request
import cv2
from flask_cors import CORS, cross_origin
import easyocr
import json
from PIL import Image
import numpy
import io
from dotenv import load_dotenv
import os
import openai
import sys
import requests

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

load_dotenv()

AUTOCORRECT_PROMPT = """
For each string in the array {}, generate 1 possible English phrase the string may have been misspelled as, using the topic of "{}" as a clue for the original phrase. Try find the most likely word for each.
Answer in as an array of JSON objects in the following format: 
[
 {{
   "possibility": <insert possible English phrase>,
   "confidence": <insert confidence as a decimal between 0 (not confident) and 1 (confident)
 }}
]"""

GPT_KEY = os.getenv('GPT_KEY')
COHERE_KEY = os.getenv('COHERE_KEY')
COHERE_KEY_2 = os.getenv('COHERE_KEY_2')
      
@app.route('/')
@cross_origin()
def hello():
    return 'Hello, World!'

@app.route("/api/text_from_image",methods=['POST'])
@cross_origin()
def text_from_image():
    THEME = request.get_json()['theme']
    BLOB = request.get_json()['blob']
    bstr = b''
    for i in range(len(BLOB)):
        bstr += BLOB[str(i)].to_bytes(1, 'big')
    BLOB = bstr

    #read image
    image = Image.open(io.BytesIO(BLOB)).convert('RGB')
    image.thumbnail((1500, 1500))
    img = numpy.array(image) 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #process image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    reader = easyocr.Reader(['en'], gpu=False)
    text = reader.readtext(thresh)
    print("image processed", file=sys.stderr)
    results = []
    texts = []
    for i in text:
        pos, phrase, confidence = i
        size = int((pos[2][0] - pos[0][0]) * (pos[2][1] - pos[0][1]))
        if size < 1000 and len(phrase) < len(text) % 75 + 2 or len(text) > 75 and len(phrase) < 7 and confidence < 0.25: # ignore small text and letters
            continue
        results.append({
            'emphasis': 0.00003 * size,
            'text': phrase,
            'confidence': float(confidence),
            'pos': str(pos),
        })
        texts.append(phrase)
    texts1 = texts[:75]
    texts2 = texts[75:150]
    texts3 = texts[150:]
    ac1 = AUTOCORRECT(texts1, THEME)#, DESC)
    ac2, ac3 = [], []
    if (len(texts2) > 0):
        ac2 = AUTOCORRECT(texts2, THEME)
        if (len(texts3) > 0):
            ac3 = AUTOCORRECT(texts3, THEME)

    ac = ac1 + ac2 + ac3
    for i in range(min(len(results), len(ac))):
        results[i]['autocorrect'] = ac[i]
    json_data = json.dumps(results, indent=2)
    print("done", file=sys.stderr)
    return json_data


def AUTOCORRECT(text, theme):#, desc):
    openai.api_key = GPT_KEY  
    try:
      response = openai.ChatCompletion.create(
          model="gpt-4",
          messages=[{"role": "user", "content": AUTOCORRECT_PROMPT.format(text, theme)}],#, desc)}],
          n=1,
          stop=None,
          temperature=0.5,
      )
      message = response.choices[0].message.content.strip()
      message = json.loads(message)
      return message
    except:
      try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5",
            messages=[{"role": "user", "content": AUTOCORRECT_PROMPT.format(text, theme)}],#, desc)}],
            n=1,
            stop=None,
            temperature=0.5,
        )
        message = response.choices[0].message.content.strip()
        message = json.loads(message)
        return message
      except:
        return []
      
@app.route("/api/generate_notes",methods=['POST'])
@cross_origin()
def generate_notes():
    topic = request.get_json()['topic']
    description = request.get_json()['description']
    data = request.get_json()['data']

    s = "Use the "
    if topic!="":
        s += f"topic \"{topic}\""
    if description!="":
        if topic!="":
            s += " and "
        s += f"description: {description}\n"
    s += "as guidelines for notes generation.\n"
    
    #write prompt here
    prompt = f"""Generate notes in point-form using the following JSON data as hints about points from a lecture/presentation. Consider the following parameters: \"emphasis\" refers to the significance of the text, so phrases with greater emphasis are more likely to be main topics/headers. Use "autocorrect.possibility" as the extracted text, and "autocorrect.confidence" refers to the probability that text is accurate. If a word with low accuracy does not fit the other subjects, it is okay to omit it.
    {s if topic!="" or description!="" else ""}
    JSON Data: {data}\n"""

    headers = {
        'Authorization': 'BEARER ' + COHERE_KEY,
        'Content-Type': 'application/json',
    }

    # model/output parameters
    json_data = {
    'model': 'command-nightly',
    "prompt": prompt,
    'max_tokens': 500,
    'temperature': 0.5,
    'k': 0,
    'stop_sequences': [],
    'return_likelihoods': 'NONE',
    }

    response = requests.post('https://api.cohere.ai/v1/generate', headers=headers, json=json_data)
    response_dict = json.loads(response.text)
    return response_dict["generations"][0]["text"]

@app.route("/api/generate_json",methods=['POST'])
@cross_origin()
def generate_json():
    open_format = open('jsonformat.txt','r')
    json_format = ""
    for line in open_format:
        json_format = json_format + line
    open_format.close()

    notes = request.get_json()['notes']

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + GPT_KEY,
    }

    json_data = {
        'model': 'gpt-4',
        'messages': [
            {
                'role': 'user',
                'content': f'Write the given text in a similar format as the given JSON. Use the markdown header information to determine what text is a header and what size the header should be. \n\nData: \n{notes} \n\nJSON: {json_format}\n',
            },
        ],
        'temperature': 0.5,
    }

    response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=json_data)
    response_dict = json.loads(response.text)
    return(response_dict['choices'][0]['message']['content'])
