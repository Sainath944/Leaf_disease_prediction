from flask import Flask, request, render_template, redirect, url_for
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import io
import base64


app = Flask(__name__)

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
model = AutoModelForImageClassification.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")


import google.generativeai as genai

import re

def clean_text(text):
    # Remove unwanted characters like '*', '\n', etc.
    cleaned_text = text.replace('*', '').replace('\n', ' ').strip()
    # Ensure there are spaces between sentences if they were joined improperly
    cleaned_text = re.sub(r'([a-zA-Z0-9])([.!?])([a-zA-Z])', r'\1\2 \3', cleaned_text)
    return cleaned_text

def format_answer(causes, prevention, about):
    # Clean the text and format it into sections with subheadings
    formatted_answer = f"### About the Disease:\n{about}\n\n"
    formatted_answer += f"### Causes of the Disease:\n{causes}\n\n"
    formatted_answer += f"### Prevention Methods:\n{prevention}\n"
    return formatted_answer

def process_text_for_all_keys(data):
    """
    takes the dictionary as input as returns a dictionary which consists of the rendered code
    """
    processed_data = {}
    
    for key, value in data.items():
        if not isinstance(value, str):  
            processed_data[key] = value
            continue
        
        lines = value.split('\n') 
        processed_lines = []
        inside_list = False  

        for line in lines:
            if line.startswith('* '):  
                if not inside_list:
                    processed_lines.append('<ul>')  
                    inside_list = True
                processed_lines.append(f"<li class='p-0'>{line[2:]}</li>")  
            else:
                if inside_list:
                    processed_lines.append('</ul>')  
                    inside_list = False
                processed_lines.append(line)  

        if inside_list:
            processed_lines.append('</ul>')  

        # html_content = '<br>'.join(processed_lines)
        html_content = ''.join(processed_lines)
        processed_data[f"{key}_html"] = html_content

    return {**data, **processed_data}


def get_details(st):
    genai.configure(api_key="use-your-own-api-key")
    # Configure the API key for Google Generative AI
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Generate content for the causes of the disease
    causes_response = model.generate_content("Causes of " + st)
    causes_ans = causes_response.text
    
    # Generate content for the prevention of the disease
    prevention_response = model.generate_content("Prevention of " + st)
    prevention_ans = prevention_response.text
    
    # Generate content for the general information about the disease
    about_response = model.generate_content("About " + st)
    about_ans = about_response.text
    
    # Combine all responses
    final_answer = {
        "Causes": causes_ans,
        "Prevention": prevention_ans,
        "About": about_ans
    }
    
    print(final_answer)
    return final_answer

#for query 
def get_response(st):
    try:
        genai.configure(api_key="use-your-own-api-key")
        # Configure the API key for Google Generative AI
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(st)
        ans = response.text
        return str(ans)
    except Exception as e:
        print(f"Error in AI model: {e}")
        return "Error generating response from the AI model."

@app.route('/')
def index():
    # Main page with upload form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded file
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
       
        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")
       
        # **NEW: Encode image as Base64 for display on the response page**
        buffered = io.BytesIO()
        image.save(buffered, format="PNG") 
        
         # Save image to an in-memory buffer
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
       
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()

        # Map the predicted class index to label
        labels = model.config.id2label
        predicted_label = labels[predicted_class_idx]
        predicted_label = get_details(predicted_label)
        predicted_label = process_text_for_all_keys(predicted_label)
        predicted_label.pop("Causes")
        predicted_label.pop("Prevention")
        predicted_label.pop("About")
        predicted_label["Causes_html"] = predicted_label['Causes_html'].replace('*', '')
        predicted_label["About_html"] = predicted_label['About_html'].replace('*', '')
        predicted_label["Prevention_html"] = predicted_label['Prevention_html'].replace('*', '')

        ### here for gemini api

        # Pass the predicted label to the result page
        return render_template('result.html', label=predicted_label, image=encoded_image)

    except Exception as e:
        return f"Error: {str(e)}"
    #for query 
@app.route('/ask-query', methods=['POST'])
def ask_query():
    try:
        # Fetch the user's query
        user_query = request.form.get('query')
        print(user_query)

        if not user_query:
            raise ValueError("Query cannot be empty.")

        # Generate response using the AI model
        response = get_response(user_query)

        # Render the response page
        return render_template('response.html', query=user_query, answer=response)

    except Exception as e:
        print(f"Error processing the query: {e}")
        return f"Error processing the query: {e}"

if __name__ == '__main__':
    app.run(debug=True)
