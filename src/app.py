from flask import Flask, request, redirect, jsonify, Blueprint, render_template, Response
from flask_cors import CORS

UPLOAD_FOLDER = 'data'

app = Flask(__name__,
			static_url_path='', 
            static_folder='static',
            template_folder='templates')

CORS(app)

#app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024