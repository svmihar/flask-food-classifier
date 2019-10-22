from app import app

@app.route('/')
@app.route('/index')
def index():
    return "<h1>Berak kuda<h1>"
