
import streamlit as st
import pickle

# Load the saved model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set up the Streamlit page
st.title('Iris Flower Classification')
st.write('This app predicts the Iris flower type based on the sepal and petal measurements.')

# Create input sliders for the features
st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.0, 8.0, 5.4)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.5, 3.4)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 7.0, 1.3)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Display the user input
user_input = user_input_features()
st.subheader('User Input:')
st.write(user_input)

# Make prediction
prediction = model.predict(user_input)
prediction_proba = model.predict_proba(user_input)

# Display the prediction
st.subheader('Prediction:')
st.write(f'The model predicts this is an **{target_names[prediction[0]]}** iris flower.')

# Display prediction probabilities
st.subheader('Prediction Probability:')
prob_df = pd.DataFrame(prediction_proba, columns=target_names)
st.write(prob_df)

# Add a sample image based on prediction
st.subheader('Iris Type:')
if prediction[0] == 0:
    st.write('Iris Setosa')
    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/440px-Kosaciec_szczecinkowaty_Iris_setosa.jpg', width=300)
elif prediction[0] == 1:
    st.write('Iris Versicolor')
    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/440px-Iris_versicolor_3.jpg', width=300)
else:
    st.write('Iris Virginica')
    st.image('https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/440px-Iris_virginica.jpg', width=300)
