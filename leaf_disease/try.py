# import absl.logging  # Import the logging library

# # Initialize logging early in your code
# absl.logging.init()

# Your code using the google-generativeai library goes here

import google.generativeai as genai

genai.configure(api_key="your-api-key")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Explain how AI works")
print(response.text)
