import os
import math
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
import sklearn.metrics
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

dataset_split_percentage = 0.8  # percentage of images for training
epochs = 80
batch_size = 100
IMG_SHAPE = 224
learningRate = 0.2

trainingDirectory: str
validationDirectory: str

def main():
    print("Hello World!")


def GetAndSaveImages():

    _URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    zip_file = tf.keras.utils.get_file(origin=_URL, fname="flower_photos.tgz", extract=True)
    base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

    classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']


    for cl in classes:
        img_path = os.path.join(base_dir, cl)
        images = glob.glob(img_path + '/*.jpg')
        print("{}: {} Images".format(cl, len(images)))
        num_train = int(round(len(images) * dataset_split_percentage))
        train, val = images[:num_train], images[num_train:]

        for t in train:
            if not os.path.exists(os.path.join(base_dir, 'train', cl)):
                os.makedirs(os.path.join(base_dir, 'train', cl))
            bn = os.path.basename(t)
            if not os.path.exists(os.path.join(base_dir, 'train', cl, bn)):
                shutil.move(t, os.path.join(base_dir, 'train', cl))

        for v in val:
            if not os.path.exists(os.path.join(base_dir, 'val', cl)):
                os.makedirs(os.path.join(base_dir, 'val', cl))
            bn = os.path.basename(v)
            if not os.path.exists(os.path.join(base_dir, 'val', cl, bn)):
                shutil.move(v, os.path.join(base_dir, 'val', cl))

        global trainingDirectory
        trainingDirectory = os.path.join(base_dir, 'train')

        global validationDirectory
        validationDirectory = os.path.join(base_dir, 'val')

        return classes


def DataGenerator():

    rotationRange = 180
    canHorizontallyFlip = True
    scale = 1. / 255

    imagesGeneratedForTraining = ImageDataGenerator(
        rescale=scale,
        rotation_range=rotationRange,
        horizontal_flip=canHorizontallyFlip,
        zoom_range=[0,0.5])

    trainDataGen = imagesGeneratedForTraining.flow_from_directory(
        batch_size=batch_size,
        directory=trainingDirectory,
        shuffle=True,
        target_size=(IMG_SHAPE, IMG_SHAPE),
        class_mode='sparse')

    imagesGeneratedForValidation = ImageDataGenerator(rescale=1. / 255)

    validationDataGen = imagesGeneratedForValidation.flow_from_directory(
        batch_size=batch_size,
        directory=validationDirectory,
        target_size=(IMG_SHAPE, IMG_SHAPE),
        class_mode='sparse')


    return trainDataGen,validationDataGen

def TrainModel(trainData, validationData, classes):


    model = tf.keras.applications.vgg19.VGG19('imagenet')
    model.summary()
    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file='VGG19.png')

    #Change Output layer
    model.trainable = False
    base_output = model.layers[-2].output  # layer number obtained from model summary above
    new_output = tf.keras.layers.Dense(5, activation="softmax")(base_output)
    model = tf.keras.models.Model(
        inputs=model.inputs, outputs=new_output)
    model.summary()

    #OPTIMIZER
    decaySteps = 10000
    decayRate = 0.7

    learningRateSchedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learningRate,
        decay_steps=decaySteps,
        decay_rate=decayRate)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learningRateSchedule)


    #Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    validationSteps = int(np.ceil(validationData.n / float(batch_size)))



    tensorboardCallback,modelCheckpointCallback,earlyStopCallback = DefineCallbacks()

    #Training time
    history = model.fit(
        trainData,
        steps_per_epoch=int(np.ceil(trainData.n / float(batch_size))),
        epochs=epochs,
        validation_data=validationData,
        validation_steps=validationSteps,
        callbacks=[tensorboardCallback,modelCheckpointCallback,earlyStopCallback]
    )

    validationLoss,validationAccuracy = model.evaluate(validationData,steps=validationSteps)
    validationX, validationY = next(validationData)

    predictionY = model.predict(validationX,steps=validationSteps)
    predictionY = np.argmax(predictionY,axis=1)
    trueY = validationY

    #Confusion Matrix
    confusionMatrix = sklearn.metrics.confusion_matrix(trueY,predictionY)


    #Other metrics
    precision = sklearn.metrics.precision_score(trueY,predictionY,average='weighted')
    recall = sklearn.metrics.recall_score(trueY,predictionY,average='weighted')
    f1 = sklearn.metrics.f1_score(trueY,predictionY,average='weighted')

   # ax,plt =
    #ax = sns.heatmap(confusionMatrix,annot=True,cmap='Blues')
    #ax.set_title("Confusion Matrix")
    #ax.xaaxis.set_ticklabels(classes)
    #ax.yaxis.set_ticklabels(classes)
    #plt.show()


    sns.make_confusion_matrix(confusionMatrix, figsize=(len(classes), len(classes)), cbar=False)

    print("Precision: {}\n".format(precision))
    print("Recall: {}\n".format(recall))
    print("F1: {}\n".format(f1))







def DefineCallbacks():

    # Tensorboard Callback
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # Checkpoint Callback
    checkpointPath = os.path.join("logs", "modelCheckpoint", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    modelCheckpointCallback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpointPath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    # Early Stopping Callback
    numberEpochsNoImprovement = 4
    metricToMeasureStopCallback = 'loss'
    earlyStopCallback = tf.keras.callbacks.EarlyStopping(monitor=metricToMeasureStopCallback,
                                                         patience=numberEpochsNoImprovement)

    return tensorboardCallback,modelCheckpointCallback,earlyStopCallback

if __name__ == "__main__":
    classes = GetAndSaveImages()
    trainDataGenerated,validationDataGenerated = DataGenerator()
    TrainModel(trainDataGenerated,validationDataGenerated,classes)
