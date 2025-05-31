from dash import Dash, html, dcc, dash_table, Input, Output
import plotly.express as px
import pandas as pd
import base64
import io


def predict_single_image_bytes(content, model_path, index_to_label):
    try:
        img = Image.open(io.BytesIO(content)).convert('RGB')
        img = img.resize((256, 256))
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - np.mean(img)) / np.std(img)
        img = np.expand_dims(img, axis=0)  # shape: (1, 256, 256, 3)

        model = tf.keras.models.load_model(model_path)
        pred = model.predict(img)
        pred_index = np.argmax(pred[0])
        return index_to_label[pred_index]
    except Exception as e:
        print(f"Error during single prediction: {str(e)}", file=sys.stderr)
        return None

# Load your saved data
df_results = pd.read_csv("C:/Users/user/OneDrive/Desktop/Big Data/lungCancer/spark/predictions.csv")
# cm = pd.read_csv("confusion_matrix.csv")  # Save CM as CSV for easier plotting
# If you do not have the 'history' object, use mock data for demonstration
history_df = pd.DataFrame({
    'Epoch': range(1, 11),
    'accuracy': [0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.8, 0.82, 0.85, 0.87],
    'loss': [0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35]
})

# Initialize Dash
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Lung Cancer Classification Dashboard", style={'textAlign': 'center'}),
    
    # 1. Training Metrics
    dcc.Graph(
        figure=px.line(history_df, x='Epoch', y=['accuracy', 'loss'], 
                      title='Model Training Performance')
    ),
    
    # 2. Confusion Matrix
    # dcc.Graph(
    #     figure=px.imshow(cm, text_auto=True,
    #                     labels=dict(x="Predicted", y="True", color="Count"),
    #                     title='Confusion Matrix')
    # ),
    
    # 3. Predictions Table
    dash_table.DataTable(
        id='predictions-table',
        data=df_results.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df_results.columns],
        page_size=10
    ),
    
    # 4. Image Upload Demo
    dcc.Upload(
        id='upload-image',
        children=html.Div(['Drag/Drop or Click to Upload']),
        style={'borderStyle': 'dashed', 'padding': '20px'}
    ),
    html.Div(id='output-image-upload'),
    html.Div(id='prediction-result')
])

# Callback for image upload
@app.callback(
    [Output('output-image-upload', 'children'),
     Output('prediction-result', 'children')],
    Input('upload-image', 'contents')
)
def update_output(content):
    if content is None:
        return None, None
    
    # Decode image
    img_str = content.split(",")[1]
    img_bytes = base64.b64decode(img_str)
    
    # Display image
    img_html = html.Img(src=content, style={'height': '256px'})
    
    # Predict (use your existing function)
    predicted_label = predict_single_image_bytes(
        img_bytes, 
        "simple_lung_cancer_model.h5", 
        index_to_label
    )
    
    result = html.H3(f"Predicted: {predicted_label}")
    
    return img_html, result

if __name__ == '__main__':
    app.run_server(debug=True)