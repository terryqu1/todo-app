from flask_cors import CORS
from flask import Flask

app = Flask(__name__)
CORS(app)

@app.route('/run-script')
def run_script():
    # Your Python code here
    return 'Script has been run!'

if __name__ == "__main__":
    app.run()

