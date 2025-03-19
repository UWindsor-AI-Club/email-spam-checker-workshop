from flask import Flask, render_template, request
import pickle 

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    user_text = ""
    if request.method == 'POST':
        user_text = request.form.get('text_input', '')
    return render_template('index.html', user_text=user_text, pred_res=run_model(user_text))


def run_model(test):
    loaded_model = pickle.load(open("spam_detector_model.pkl", 'rb'))
    vectorizer = pickle.load(open("spam_detector_vectorizer.pkl", 'rb'))

    text_features = vectorizer.transform([test])
    res = loaded_model.predict(text_features)
    if(res == 0):
        return "The email is not spam"
    else:
        return "The email is spam"
    return "I don't know"

if __name__ == '__main__':
    app.run(debug=True)
