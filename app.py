from flask import Flask,request,jsonify,render_template
from flask_cors import *

from util import process_source, get_ast
from CCGIR import Retrieval

ccgir = Retrieval()

print("Sentences to vectors")
ccgir.encode_file()

print("加载索引")
ccgir.build_index(n_list=1)
ccgir.index.nprob = 1

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
CORS(app, supports_credentials=True)

from flask.json import JSONEncoder as _JSONEncoder

class JSONEncoder(_JSONEncoder):
    def default(self, o):
        import decimal
        if isinstance(o, decimal.Decimal):
            return float(o)
        super(JSONEncoder, self).default(o)
app.json_encoder = JSONEncoder

@app.route('/predict',methods=['GET'])
def predict():
    origin_code = str(request.args['code'])
    code = process_source(origin_code)
    ast = get_ast(origin_code)
    sim_code, sim_ast, sim_nl = ccgir.single_query(code, ast, topK=5)
    result = {'nl':sim_nl}
    return jsonify(result)

@app.route('/index', methods=["GET"])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
