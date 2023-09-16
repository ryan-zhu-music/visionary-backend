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


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

load_dotenv()

AUTOCORRECT_PROMPT = """
For each string in the array {}, generate 1 possible English phrase the string may have been misspelled as, using the theme of {} as a clue for the original phrase. Try to maximize the confidence of each phrase.
Answer in as an array of JSON objects in the following format: 
[
 {{
   "possibility": <insert possible English phrase>,
   "confidence": <insert confidence as a decimal between 0 (not confident) and 1 (confident)
 }}
]"""

GPT_KEY = os.getenv('GPT_KEY')
      
@app.route('/')
@cross_origin()
def hello():
    return 'Hello, World!'

@app.route("/api/text_from_image",methods=['POST'])
@cross_origin()
def text_from_image():
    if (request.method != 'POST'):
        return "Error: Invalid request method"
    
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

    results = []
    texts = []
    for i in text:
        pos, phrase, confidence = i
        size = int((pos[2][0] - pos[0][0]) * (pos[2][1] - pos[0][1]))
        if size < 1000 and len(phrase) < len(text) % 75 + 2 or len(text) > 75 and len(phrase) < 7 and confidence < 0.25: # ignore small text and letters
            continue
        results.append({
            'size': size,
            'text': phrase,
            'confidence': float(confidence),
            'pos': str(pos),
        })
        texts.append(phrase)
    texts1 = texts[:75]
    texts2 = texts[75:150]
    texts3 = texts[150:]
    ac1 = AUTOCORRECT(texts1, THEME)
    ac2, ac3 = [], []
    if (len(texts2) > 0):
        ac2 = AUTOCORRECT(texts2, THEME)
        if (len(texts3) > 0):
            ac3 = AUTOCORRECT(texts3, THEME)

    ac = ac1 + ac2 + ac3
    for i in range(min(len(results), len(ac))):
        results[i]['autocorrect'] = ac[i]
    results.sort(key=lambda x: int(x['size']), reverse=True)
    json_data = json.dumps(results, indent=2)

    return json_data
def AUTOCORRECT(text, theme):
    openai.api_key = GPT_KEY  
    try:
      response = openai.ChatCompletion.create(
          model="gpt-4",
          messages=[{"role": "user", "content": AUTOCORRECT_PROMPT.format(text, theme)}],
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
            messages=[{"role": "user", "content": AUTOCORRECT_PROMPT.format(text, theme)}],
            n=1,
            stop=None,
            temperature=0.5,
        )
        message = response.choices[0].message.content.strip()
        message = json.loads(message)
        return message
      except:
        return []