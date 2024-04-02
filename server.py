from flask import Flask, request, render_template, send_from_directory, flash  # Import flask
import os
from werkzeug.utils import secure_filename

# ------------------------------
# Environment Variable Setup
# ------------------------------
from dotenv import load_dotenv

# Load the base .env file
load_dotenv(".env")

# for cnn
from torchvision.io import read_image
import torch
from torchvision import transforms
from cnn.models import Identicycle
from cnn.models import Identicycle_Filters
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model
model_name = "3x3-3L-128N-200E-Model1"
model_path = os.path.join(os.path.dirname( __file__ ), 'cnn', 'models', model_name)
model_Test = Identicycle(input_shape=3, hidden_units=128, output_shape=7)
model_Test.load_state_dict(torch.load(f=model_path, map_location=torch.device('cpu')))
model_Filters = Identicycle_Filters(input_shape=3, hidden_units=128, output_shape=7)

# WSGI Application
# Defining upload folder path
UPLOAD_FOLDER = os.path.join('static', 'uploads')
# # Define allowed files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__, static_folder='static', static_url_path='/', template_folder='static')  # Setup the flask app by creating an instance of Flask

# Configure upload folder for Flask application
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image Selected to be used for the inner node visualization
image_selected = [[2,3,4,6,8,10,12,14],[2,4,6,9,12,14,15,20,21,24,26,27,28,29,32,33],[2,3,7,9,10,12,13,18,19,25,26,27,29,30,31,32,33,35,38,41,42,46,50,51,53,54,55,56,58,60,62,64,65,68,69,501]]

@app.route('/')  # When someone goes to / on the server, execute the following function
def home():
  return app.send_static_file('index.html')  # Return index.html from the static folder

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def load_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image)
        return image_tensor
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


@app.route('/predict',  methods=("POST", "GET"))
def image_upload_view():
    if request.method == 'POST':
        if 'uploaded-image' not in request.files:
            flash("no file part")
            return "No file part"
        uploaded_img = request.files['uploaded-image']
        
        if uploaded_img.filename == '' or not allowed_file(uploaded_img.filename):
            flash("No selected file or file type not allowed")
        img_filename = secure_filename(uploaded_img.filename)
        img_name = img_filename.split(".")[0]
        image_path = os.path.join(os.path.dirname(__file__), app.config['UPLOAD_FOLDER'], img_filename)
        uploaded_img.save(image_path)
        
        image_tensor = load_image(image_path)
        if image_tensor is None:
           return "Error processing image. Please upload a valid image file."
    # Make predictions
    model_Test.eval()
    with torch.inference_mode():
        custom_image_pred = model_Test(image_tensor.unsqueeze(dim=0))
    custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
    custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1).item()

    # Define class names
    class_names = ['ewaste', 'glass', 'metal', 'organic', 'paper', 'plastic', 'trash']
    custom_image_pred_class = class_names[custom_image_pred_label]
    custom_image_pred_probs = custom_image_pred_probs.squeeze(dim=0)
    probabilities = custom_image_pred_probs

    get_Image_Filters(image_tensor.unsqueeze(dim=0),img_name, image_tensor)

    data = {
        'image_name': img_name,
        'image_filename': img_filename,
        'image_path': image_path,
        'image_selected': image_selected,
        'predicted_class': custom_image_pred_class,
        'predicted_probs': {class_names[i]: round(probabilities[i].item(),4) for i in range(len(class_names))}
    }
    return render_template("result.html", data = data)

def get_Image_Filters(img,name,raw_img):

# Forward pass through the model to get activation maps
  model_Filters.eval()
  with torch.inference_mode():
    outputs, activation_maps = model_Filters(img)
# Create a directory to save the activation map images
  output_dir = 'static/uploads/conv_images'
  os.makedirs(output_dir, exist_ok=True)
  # raw_img_path = os.path.join(output_dir, f'{name}_0_0_image.png')
  # plt.imsave(raw_img_path, raw_img.toTensor().permute(2,1,0))
# Save each activation map image individually
  for i, maps in enumerate(activation_maps):
    num_filters = maps.size(1)  # Number of filters
    for j in range(num_filters):
      if j in image_selected[i]: # to save only images needed
        filter_path = os.path.join(output_dir, f'{name}_{i+1}_{j}_image.png')
        filter_image = maps[0, j].detach().cpu().numpy()
        plt.imsave(filter_path, filter_image)
if __name__ == '__main__':  # If the script that was run is this script (we have not been imported)
  if os.environ.get('ENV')=="development":
    app.run(host="localhost", port=8080, debug=True)
  else:
    app.run(host="0.0.0.0", port=80)
    