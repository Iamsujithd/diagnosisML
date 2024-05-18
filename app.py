from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/Symptom')
def symptoms():
    return render_template('Symptom.html')

if __name__ == '__main__':
    app.run()
from flask import request

@app.route('/signin', methods=['POST'])
def signin():
    username = request.form['username']
    password = request.form['password']
    # Add your sign-in logic here
  
