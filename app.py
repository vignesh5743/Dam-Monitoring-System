from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, r2_score

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('wallcrackdetection.h5')

# Function to generate a DataFrame from an image directory
def generate_df(img_dir, label):
    file_paths = pd.Series(list(img_dir.glob('*.jpg')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=file_paths.index)
    df = pd.concat([file_paths, labels], axis=1)
    return df

# Your existing code for data generation and model setup goes here
positive_dir = Path('Positive')
negative_dir = Path('Negative')

# ... (Your existing code)
def generate_df(img_dir, label):
    
    file_paths = pd.Series(list(img_dir.glob(r'*.jpg')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=file_paths.index)
    df = pd.concat([file_paths, labels], axis=1)
    
    return df
	
	
	
positive_df = generate_df(positive_dir, 'POSITIVE')
negative_df = generate_df(negative_dir, 'NEGATIVE')

# concatenate both positive and negative df
all_df = pd.concat([positive_df, negative_df], axis=0).sample(frac=1, random_state=1).reset_index(drop=True)
all_df

train_df, test_df = train_test_split(all_df.sample(6000, random_state=1), 
                train_size=0.7,
                shuffle=True,
                random_state=1)
				
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                           validation_split=0.2)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_dataframe(train_df, 
                                          x_col='Filepath',
                                          y_col='Label',
                                          target_size=(120,120), 
                                          color_mode='rgb',
                                          class_mode='binary',
                                          batch_size=32,
                                          shuffle=True,
                                          seed=42,
                                          subset='training')


val_data = train_gen.flow_from_dataframe(train_df, 
                                          x_col='Filepath',
                                          y_col='Label',
                                          target_size=(120,120), 
                                          color_mode='rgb',
                                          class_mode='binary',
                                          batch_size=32,
                                          shuffle=True,
                                          seed=42,
                                          subset='validation')


test_data = test_gen.flow_from_dataframe(test_df, 
                                          x_col='Filepath',
                                          y_col='Label',
                                          target_size=(120,120), 
                                          color_mode='rgb',
                                          class_mode='binary',
                                          batch_size=32,
                                          shuffle=False,
                                          seed=42)
										  
										  
inputs = tf.keras.Input(shape=(120,120,3))
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
			 
			 
history = model.fit(train_data, validation_data=val_data, epochs=100, 
                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              patience=3,
                                                              restore_best_weights=True)
                             ])
							 
							 
fig = px.line(history.history,
             y=['loss', 'val_loss'],
             labels={'index':'Epoch'},
             title='Training and Validation Loss over Time')

fig.show()


def evaluate_model(model, test_data):
    
    results = model.evaluate(test_data, verbose=0)
    loss = results[0]
    accuracy = results[1]
    
    print(f'Test Loss {loss:.5f}')
    print(f'Test Accuracy {accuracy * 100:.2f} %')
    
    
    # predicted y values
    y_pred = np.squeeze((model.predict(test_data) >= 0.5).astype(int))
    y_certain = np.squeeze((model.predict(test_data)).astype(int))
    
    conf_matr = confusion_matrix(test_data.labels, y_pred)
    
    class_report = classification_report(test_data.labels, y_pred,
                                         target_names=['NEGATIVE', 'POSITIVE'])
    
    plt.figure(figsize=(6,6))
    
    sns.heatmap(conf_matr, fmt='g', annot=True, cbar=False, vmin=0, cmap='Blues')
    
    plt.xticks(ticks=np.arange(2) + 0.5, labels=['NEGATIVE', 'POSITIVE'])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=['NEGATIVE', 'POSITIVE'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    print('r2 Score : ', r2_score(test_data.labels, y_pred))
    print()
    print('Classification Report :\n......................\n', class_report)
	
def test_new_data(dir_path):
    
    new_test_dir = Path(dir_path)
    
    df_new = generate_df(new_test_dir, 'Testing')
    
    test_data_new = test_gen.flow_from_dataframe(df_new, 
                                          x_col='Filepath',
                                          y_col='Label',
                                          target_size=(120,120), 
                                          color_mode='rgb',
                                          batch_size=5,
                                          shuffle=False,
                                          seed=42)
    
        # predicted y values
    y_pred = np.squeeze((model.predict(test_data_new) >= 0.5).astype(int))
    
    
    y_certain = model.predict(test_data_new).round(6)
    y_out = []
    for i in y_pred:
        if i==0:
            y_out.append('Negative (Not Crack)')
        else:
            y_out.append('Positive(Crack) ')
            
    result = pd.DataFrame(np.c_[y_out, y_certain], columns=['Result', 'Confidance of being Cracked'])
    
    return result

# Define a route to capture an image
@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        # Capture and save an image
        capture_image()

        # Load and process the captured image
        captured_image = cv2.imread('captured_image.jpg')
        captured_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
        captured_image = cv2.resize(captured_image, (120, 120))
        captured_image = captured_image / 255.0

        # Predict the label for the captured image
        y_pred = model.predict(np.expand_dims(captured_image, axis=0))
        result = "Positive (Crack)" if y_pred >= 0.5 else "Negative (Not Crack)"
        confidence = f"Confidence: {y_pred[0][0]:.2f}"

        return render_template('result.html', result=result, confidence=confidence)

    return render_template('capture.html')

if __name__ == '__main__':
    app.run(debug=True)
