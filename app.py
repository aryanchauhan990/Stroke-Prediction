import pickle
from flask import Flask , url_for , request , jsonify , render_template
import numpy as np

app = Flask(__name__)
filename = "LGBMClassifiernew.sav"
model=pickle.load(open(filename,'rb'))

@app.route("/")
def home():
    return(render_template('index.html'))



@app.route('/predict',methods=['POST'])
def predict():
    features = [ float(x) for x in request.form.values()]
    
    #Logic for Work Type becuase I have used one hot encoding
    lis_work=[]
    lis_smoke=[]
    if(features[-2]==0):
        lis_work=[1,0,0,0,0]
    elif(features[-2]==1):
        lis_work=[0,1,0,0,0]
    elif(features[-2]==2):
        lis_work=[0,0,1,0,0]
    elif(features[-2]==3):
        lis_work=[0,0,0,1,0]
    elif(features[-2]==4):
        lis_work=[0,0,0,0,1]


    #Logic for Smoking Status becuase I have used one hot encoding
    if(features[-1]==0):
        lis_smoke=[1,0,0,0]
    elif(features[-1]==1):
        lis_smoke=[0,1,0,0]
    elif(features[-1]==2):
        lis_smoke=[0,0,1,0]
    elif(features[-1]==3):
        lis_smoke=[0,0,0,1]



    features = features[:-2]+lis_work+lis_smoke
    print(features)
    f_features = [np.array(features)]

    prediction = model.predict(f_features)
    print(prediction)
    if(prediction==0):
         return render_template('index.html', prediction_text='Patient is not likely to have a stroke')

    elif(prediction==1):
        return render_template('index.html', prediction_text='Patient is likely to have a stroke')



if __name__ == "__main__":
    app.run(debug=True)

    
    

