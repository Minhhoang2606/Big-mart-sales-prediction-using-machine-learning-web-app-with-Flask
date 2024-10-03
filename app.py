from flask import Flask, request, render_template
import numpy as np
import pickle
import sklearn

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Item_Identifier = request.form['Item_Identifier']
        Item_Weight = request.form['Item_Weight']
        Item_Fat_Content = request.form['Item_Fat_Content']
        Item_Visibility = request.form['Item_Visibility']
        Item_Type = request.form['Item_Type']
        Item_MRP = request.form['Item_MRP']
        Outlet_Identifier = request.form['Outlet_Identifier']
        Outlet_Establishment_Year = request.form['Outlet_Establishment_Year']
        Outlet_Size = request.form['Outlet_Size']
        Outlet_Location_Type = request.form['Outlet_Location_Type']
        Outlet_Type = request.form['Outlet_Type']

        features = np.array([[Item_Identifier,Item_Weight,Item_Fat_Content,Item_Visibility,Item_Type,Item_MRP,Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type,Outlet_Type]],dtype=np.float32)
        transformed_feature = features.reshape(1, -1)
        prediction = model.predict(transformed_feature)[0]
        print(prediction)
        return render_template('index.html', prediction=prediction)
    return render_template('index.html', prediction="Error in prediction")
    

if __name__=="__main__":
    app.run(debug=True)