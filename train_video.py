from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


TRAIN_DIR = r"D:\ulm\Course Cogsys\Sem 4\AI for Auto\images\images\train"
TEST_DIR = r"D:\ulm\Course Cogsys\Sem 4\AI for Auto\images\images\validation"

def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir,label)):
            image_paths.append(os.path.join(dir,label,imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths,labels


def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, grayscale=True)
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features


train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)
test = pd.DataFrame()
test['image'], test['label'] = createdataframe(TEST_DIR)
train_features = extract_features(train['image'])
test_features = extract_features(test['image'])
x_train = train_features/255.0 #Dividing by 255 converts them to a floating-point range between 0.0 and 1.0.
x_test = test_features/255.0

le = LabelEncoder()
le.fit(train['label']) #This scans all unique emotion labels (7 emotions) and assigns each one a number

y_train = le.transform(train['label'])
y_test = le.transform(test['label'])

y_train = to_categorical(y_train,num_classes = 7) #Label	One-hot encoded vector
                                                  #0(angry)	[1, 0, 0, 0, 0, 0, 0]
y_test = to_categorical(y_test,num_classes = 7)

model = Sequential()
# convolutional layers
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1))) # Extract visual features and relu = max(0, x)
model.add(MaxPooling2D(pool_size=(2,2))) #Shrinks the image and keeps only strong features
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Flatten())
# fully connected layers
model.add(Dense(512, activation='relu')) #If convolution layers are your “feature extractors,” Dense layers are your “decision-makers.”
model.add(Dropout(0.4))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
# output layer
model.add(Dense(7, activation='softmax'))


model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )

#model.fit(x= x_train,y = y_train, batch_size = 128, epochs = 100, validation_data = (x_test,y_test))
# Save the best model during training
checkpoint = ModelCheckpoint(
    'train_model.keras',        # filename to save
    monitor='val_loss',     # metric to monitor
    save_best_only=True,    # only save when val_loss improves
    mode='min',             # 'min' because lower loss is better
    verbose=1
)

# Stop training early if no improvement
early_stop = EarlyStopping(
    monitor='val_loss',     # metric to monitor
    patience=5,             # epochs to wait before stopping
    restore_best_weights=True,
    verbose=1
)
# Train model with callbacks
history = model.fit(
    x=x_train, y=y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test),
    callbacks=[checkpoint, early_stop]
)

# Predict probabilities for the test set
y_pred_probs = model.predict(x_test)

# Convert predicted probabilities to class labels
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Video Emotion Model')
plt.show()

report = classification_report(y_true_classes, y_pred_classes, target_names=le.classes_)
print(report)


acc = accuracy_score(y_true_classes, y_pred_classes)
prec = precision_score(y_true_classes, y_pred_classes, average='weighted')
rec = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")

import random
indices = random.sample(range(len(x_test)), 1)
for i in indices:
    plt.imshow(x_test[i].reshape(48,48), cmap='gray')
    plt.title(f"True: {le.inverse_transform([y_true_classes[i]])[0]}, Pred: {le.inverse_transform([y_pred_classes[i]])[0]}")
    plt.axis('off')
    plt.show()

# Loss, Accuracy presentation

# Plot history: Loss
plt.plot(history.history['loss'], label='Loss (training data)')
plt.plot(history.history['val_loss'], label='Loss (validation data)')
plt.title('Loss for train and validation')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()


model_json = model.to_json()
with open("emotiondetector.json",'w') as json_file:
    json_file.write(model_json)
model.save("emotiondetector.keras")