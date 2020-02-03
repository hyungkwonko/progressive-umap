from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def result():
    return render_template('result.html')
    
if __name__ == "__main__":
    app.run() # FLASK_APP=app.py python -m flask run