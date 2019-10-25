# from app import app

from flask import Flask

app = Flask(__name__)


@app.route('/hi/<str:name>')
def hi(name):
    return f'Hello {name}'


if __name__ == "__main__":
    app.run(debug=True)