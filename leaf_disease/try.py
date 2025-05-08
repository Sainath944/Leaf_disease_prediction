# import absl.logging  # Import the logging library

# # Initialize logging early in your code
# absl.logging.init()

# Your code using the google-generativeai library goes here

import google.generativeai as genai

genai.configure(api_key="AIzaSyAS7UP1sNxVjJCXm59s22zPvByipL1f6Mg")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Explain how AI works")
print(response.text)