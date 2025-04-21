from flask import Flask
from flask_cors import CORS
from flask import render_template
app = Flask(__name__)
CORS(app)

@app.route('/',methods=['POST','GET'])
def serve_home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)