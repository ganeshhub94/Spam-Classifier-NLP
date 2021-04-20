import pickle
from flask import render_template,url_for,request,Flask

file1 = open('model_pickle','rb')
file2= open('transform_pkl1','rb')

model = pickle.load(file1)
tranformer = pickle.load(file2)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        preds = tranformer.transform(data).toarray()
        mypred = model.predict(preds)
    return render_template ('result.html',prediction = mypred)
    

if __name__ == "__main__":
    app.run(debug = True)

