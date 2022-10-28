from audioop import cross
from flask import Flask
from flask import request
import vk
from chatbot import predict_answer
from flask_cors import CORS, cross_origin


api = vk.API(access_token='9c3ce37a9c3ce37a9c3ce37a929f2dca2b99c3c9c3ce37aff6811541c8a4debfafab744', v='5.131')



app = Flask(__name__)
CORS(app)

@app.route('/vk-main-post', methods=['GET'])
@cross_origin()
def main():
    raw = api.wall.get(owner_id = -47535294, count = 1)
    content = raw['items'][0]['text']
    
    return content

@app.route('/vk-sic-post', methods=['GET'])
@cross_origin()
def sic():
    raw = api.wall.get(owner_id = -76527561, count = 1)
    content = raw['items']
    return content

@app.route('/vk-com-post', methods=['GET'])
@cross_origin()
def com():
    raw = api.wall.get(owner_id = 485005212, count = 1)
    content = raw['items'][0]['text']
    return content

@app.route('/vk-abit-post', methods=['GET'])
@cross_origin()
def abit():
    raw = api.wall.get(owner_id = -205247745, count = 1)
    content = raw['items'][0]['text']
    return content
    

@app.route('/', methods=['POST'])
@cross_origin()
def post():
    content = request.json['content']
    answer = predict_answer(content)
    return answer