from flask import Flask, render_template, request
import re

app = Flask(__name__)

def extract_ph_potassium(text):
    # Define regular expressions for pH and potassium values
    ph_pattern = re.compile(r'pH:\s*(\d+(\.\d+)?)')
    potassium_pattern = re.compile(r'Potassium:\s*(\d+(\.\d+)?)')

    # Extract pH and potassium values from the text
    ph_match = ph_pattern.search(text)
    potassium_match = potassium_pattern.search(text)

    # Perform basic prediction based on extracted values
    if ph_match:
        ph_value = float(ph_match.group(1))
        if ph_value > 7.0:
            ph_prediction = "Alkaline"
        else:
            ph_prediction = "Acidic"
    else:
        ph_prediction = "Not available"

    if potassium_match:
        potassium_value = float(potassium_match.group(1))
        if potassium_value > 10.0:
            potassium_prediction = "High"
        else:
            potassium_prediction = "Normal"
    else:
        potassium_prediction = "Not available"

    return ph_prediction, potassium_prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the POST request has a file part
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        # Check if the file is a text file
        if file and file.filename.endswith('.txt'):
            text_content = file.read().decode('utf-8')  # Read text file content

            # Extract and predict pH and potassium values
            ph_prediction, potassium_prediction = extract_ph_potassium(text_content)

            return render_template('index.html', ph_prediction=ph_prediction, potassium_prediction=potassium_prediction)
        else:
            return render_template('index.html', error='Invalid file format. Please upload a text file.')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
