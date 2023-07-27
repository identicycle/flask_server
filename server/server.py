from flask import Flask, request, render_template, send_from_directory  # Import flask
import os
from werkzeug.utils import secure_filename

# for cnn
from torchvision.io import read_image
import torch
from torchvision import transforms
from models import Identicycle

# Load the trained model
model_name = "3x3-3L-128N-200E-Model1"
model_path = os.path.join(os.path.dirname( __file__ ), 'models', model_name)
model_Test = Identicycle(input_shape=3, hidden_units=128, output_shape=7)
model_Test.load_state_dict(torch.load(f=model_path, map_location=torch.device('cpu')))

# WSGI Application
# Defining upload folder path
UPLOAD_FOLDER = os.path.join('..', 'static', 'uploads')
# # Define allowed files
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__, static_folder='../static', static_url_path='/', template_folder='../static')  # Setup the flask app by creating an instance of Flask

# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'

@app.route('/')  # When someone goes to / on the server, execute the following function
def home():
  return app.send_static_file('index.html')  # Return index.html from the static folder

@app.route('/predict',  methods=("POST", "GET"))
def image_upload_view():
  if request.method == 'POST':
    uploaded_img = request.files['uploaded-image']
    # Extracting uploaded data file name
    img_filename = secure_filename(uploaded_img.filename)
    # Upload file to database (defined uploaded folder in static path)
    image_path = os.path.join(os.path.dirname( __file__ ), app.config['UPLOAD_FOLDER'], img_filename)
    uploaded_img.save(image_path)
    
    # Read the image from the temporary file and preprocess it
    custom_image = read_image(image_path).type(torch.float32) / 255.
    custom_image_transform = transforms.Compose([
        transforms.Resize((128, 128)),
    ])
    custom_image_transformed = custom_image_transform(custom_image)

    # Make predictions
    model_Test.eval()
    with torch.inference_mode():
        custom_image_pred = model_Test(custom_image_transformed.unsqueeze(dim=0))
    custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
    custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1).item()

    # Define class names
    class_names = ['ewaste', 'glass', 'metal', 'organic', 'paper', 'plastic', 'trash']
    custom_image_pred_class = class_names[custom_image_pred_label]
    custom_image_pred_probs = custom_image_pred_probs.squeeze(dim=0)
    probabilities = custom_image_pred_probs

    data = {
        'image_name': img_filename,
        'image_path': image_path,
        'predicted_class': custom_image_pred_class,
        'predicted_probs': {class_names[i]: round(probabilities[i].item(),4) for i in range(len(class_names))}
    }
    return render_template("result.html", data = data)

  return "Please upload proper image"

if __name__ == '__main__':  # If the script that was run is this script (we have not been imported)
  app.run(host="localhost", port=8080, debug=True)  # Start the server