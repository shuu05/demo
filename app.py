import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.array_to_img(img)
    expanded_img_array =np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result/norm(result)

    return  normalized_result


filename = []
for file in os.listdir('images'):
    filename.append(os.path.join('images', file))


#print(len(filename))
#print(filename[0:5])

feature_list = []

for file in tqdm(filename):
    feature_list.append(extract_features(file,model))

pickle.dump(feature_list, open('venv/embeddings.pkl', 'wb'))
pickle.dump(filename, open('venv/filenames.pkl', 'wb'))

#print(model.summary())
