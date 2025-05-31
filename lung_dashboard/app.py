import os
import base64
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from io import BytesIO
from dash import Dash, html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.express as px
from sklearn.manifold import TSNE

# Predefined confusion matrix
conf_matrix = np.array([
    [20, 32, 0, 26],
    [3, 10, 0, 5],
    [0, 1, 43, 1],
    [4, 21, 0, 34]
])

# Dataset paths
DATASET_PATHS = {
    "train": "uploads/train",
    "valid": "uploads/valid",
    "test": "uploads/test"
}

# Load trained model
model = tf.keras.models.load_model("simple_lung_cancer_model.h5")

# Build the model explicitly
model.build(input_shape=(None, 256, 256, 3))
dummy_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
_ = model.predict(dummy_input)

# Create feature extraction model
layer_output = model.layers[-2].output
input_tensor = model.inputs[0]
feature_model = tf.keras.Model(inputs=input_tensor, outputs=layer_output)

# Class labels
class_names = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']

# Create confusion matrix visualization
df_cm = pd.DataFrame(
    conf_matrix,
    index=class_names,
    columns=class_names
)

conf_matrix_fig = px.imshow(
    conf_matrix,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=class_names,
    y=class_names,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='Blues'
)

conf_matrix_fig.update_layout(
    title='Confusion Matrix - Lung Cancer Classification',
    xaxis_title='Predicted Label',
    yaxis_title='True Label',
    font=dict(size=12),
    height=600,
    width=800
)

# Add annotations to each cell
for i in range(len(class_names)):
    for j in range(len(class_names)):
        conf_matrix_fig.add_annotation(
            x=j, y=i,
            text=str(conf_matrix[i, j]),
            showarrow=False,
            font=dict(
                color='white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black',
                size=14
            )
        )

# Preprocess uploaded image
def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = (img_array - np.mean(img_array)) / np.std(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Dataset analysis (class distribution)
def analyze_dataset_structure(base_dirs):
    data = []
    for split_name, path in base_dirs.items():
        for class_name in os.listdir(path):
            class_path = os.path.join(path, class_name)
            if os.path.isdir(class_path):
                image_count = len([f for f in os.listdir(class_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
                data.append({
                    "Dataset": split_name.capitalize(),
                    "Class": class_name.replace('_', ' ').capitalize(),
                    "Images": image_count
                })
    return pd.DataFrame(data)

# Average image per class
def compute_average_image_per_class(base_path, class_names, size=(256, 256)):
    avg_images = {}
    for cls in class_names:
        cls_path = os.path.join(base_path, cls.lower().replace(" ", "_"))
        images = []
        for f in os.listdir(cls_path):
            if f.lower().endswith(('jpg', 'png', 'jpeg')):
                img = Image.open(os.path.join(cls_path, f)).convert('RGB')
                img = img.resize(size)
                images.append(np.array(img).astype(np.float32))
        if images:
            mean_img = np.mean(images, axis=0).astype(np.uint8)
            avg_images[cls] = Image.fromarray(mean_img)
    return avg_images

# Visualize latent space using t-SNE
def visualize_latent_space(images, labels, model):
    feature_model = tf.keras.Model(inputs=model.inputs[0], outputs=model.layers[-2].output)
    features = feature_model.predict(images)
    n_samples = features.shape[0]
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1

    if n_samples <= 1:
        return px.scatter(title="Not enough samples for t-SNE")

    reduced = TSNE(n_components=2, perplexity=perplexity).fit_transform(features)
    fig = px.scatter(x=reduced[:, 0], y=reduced[:, 1], color=labels, title="Latent Feature Clustering (t-SNE)")
    return fig

# App setup
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Lung Cancer MRI Classifier"

# Layout
app.layout = dbc.Container([
    html.H1("🧬 Lung Cancer MRI Classifier", className="text-center mt-4"),
    
    # Classification Section
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-image',
                children=html.Div(['📁 Drag & Drop or ', html.A('Select an MRI Image')]),
                style={
                    'width': '100%',
                    'height': '150px',
                    'lineHeight': '150px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='output-prediction'),
        ], width=6),
        
        dbc.Col([
            html.Div(id='output-image'),
        ], width=6)
    ]),
    
    html.Hr(),
    html.H2("📊 Dataset Overview", className="mt-4"),
    dcc.Graph(id='class-distribution'),
    
    html.Hr(),
    html.H2("🧠 Average Image Per Class"),
    html.Div(id='average-image-display', className="d-flex flex-wrap gap-3"),
    
    html.Hr(),
    html.H2("🌌 Feature Clustering (t-SNE)"),
    dcc.Graph(id='latent-space-plot'),
    
    # Moved Confusion Matrix to the End
    html.Hr(),
    html.H2("📈 Model Performance Metrics", className="mt-4"),
    dbc.Card([
        dbc.CardBody([
            html.H4("Confusion Matrix", className="card-title"),
            dcc.Graph(figure=conf_matrix_fig)
        ])
    ]),
], fluid=True)

# Prediction callback
@app.callback(
    Output('output-image', 'children'),
    Output('output-prediction', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename')
)
def classify_image(contents, filename):
    if contents is None:
        return None, None

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = Image.open(BytesIO(decoded)).convert('RGB')

    img_display = html.Img(src=contents, style={'maxWidth': '100%', 'height': 'auto'})

    img_array = preprocess_image(image)
    preds = model.predict(img_array)[0]
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)

    result = dbc.Alert(
        f"Prediction: {pred_class} ({confidence*100:.2f}%)",
        color="info",
        className="mt-3"
    )

    return img_display, result

# Dataset distribution bar chart
@app.callback(
    Output('class-distribution', 'figure'),
    Input('upload-image', 'contents')
)
def update_dataset_analysis(_):
    df = analyze_dataset_structure(DATASET_PATHS)
    fig = px.bar(df, x="Class", y="Images", color="Dataset", barmode="group",
                 title="Image Count per Class in Each Dataset Split")
    return fig

# Average image per class
@app.callback(
    Output('average-image-display', 'children'),
    Input('upload-image', 'contents')
)
def update_average_images(_):
    avg_images = compute_average_image_per_class(DATASET_PATHS["train"], class_names)
    img_elements = []
    for label, image in avg_images.items():
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()
        img_elements.append(
            html.Div([
                html.H6(label, className="text-center"),
                html.Img(src=f"data:image/png;base64,{encoded}", 
                         style={"height": "150px", "margin": "5px", "border": "1px solid #ddd"})
            ], className="text-center p-2")
        )
    return img_elements

# Latent space plot
@app.callback(
    Output('latent-space-plot', 'figure'),
    Input('upload-image', 'contents')
)
def update_latent_plot(_):
    images = []
    labels = []
    for cls in class_names:
        folder = os.path.join(DATASET_PATHS["train"], cls.lower().replace(" ", "_"))
        if not os.path.exists(folder):
            print(f"Folder does not exist: {folder}")
            continue
        files = [f for f in os.listdir(folder) if f.lower().endswith(('jpg', 'png', 'jpeg'))][:5]
        for f in files:
            try:
                img = Image.open(os.path.join(folder, f)).convert("RGB").resize((256, 256))
                img = np.array(img).astype(np.float32) / 255.0
                img = (img - np.mean(img)) / (np.std(img) + 1e-7)
                images.append(img)
                labels.append(cls)
            except Exception as e:
                print(f"Error loading image {f} from {folder}: {e}")

    if not images:
        return px.scatter(title="No images available")

    images_np = np.stack(images)
    
    try:
        fig = visualize_latent_space(images_np, labels, model)
    except Exception as e:
        print(f"Error in visualize_latent_space: {e}")
        fig = px.scatter(title="Error generating latent space plot")
    return fig

# Run app
if __name__ == '__main__':
    app.run(debug=True)