import os
import cv2
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
import random as r
import skimage.transform as skimage_transform
import skimage.io as skimage_io
import PIL.Image as PImage

dataset_split_percentage = 0.8  # percentage of images for training
epochs = 80
batchSize = 32
IMG_SHAPE = 229
inputSize = (IMG_SHAPE, IMG_SHAPE)
learningRate = 0.1

dataSetDirectory :str
trainingDirectory: str
validationDirectory: str
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
classesDict= dict(roses=0,daisy=1,dandelion=2,sunflowers=3,tulips=4)


testFolder = "test"
modelsPath = "models"

trainSize = -1  # -1 for all
valSize = -1  # -1 for all
testSize = -1  # -1 for all

augmentation_args = dict(
    width_shift_range=range((int)(IMG_SHAPE/2)),
    height_shift_range=range((int)(IMG_SHAPE/2)),
    horizontal_flip=True,
    zoom=[True,0.5]
)





def DownloadAndSaveImages():

    _URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    zip_file = tf.keras.utils.get_file(origin=_URL, fname="flower_photos.tgz", extract=True)
    base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

    counter = 0
    for cl in classes:


        #Store in different pastes
        img_path = os.path.join(base_dir, cl)
        images = glob.glob(img_path + '/*.jpg')
        #print("{}: {} Images".format(cl, len(images)))

        #Separate for train and val
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


        global dataSetDirectory
        dataSetDirectory = base_dir

        global trainingDirectory
        trainingDirectory = os.path.join(base_dir, 'train')

        global validationDirectory
        validationDirectory = os.path.join(base_dir, 'val')




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
        batch_size=batchSize,
        directory=trainingDirectory,
        shuffle=True,
        target_size=(IMG_SHAPE, IMG_SHAPE),
        class_mode='sparse')

    imagesGeneratedForValidation = ImageDataGenerator(rescale=1. / 255)

    validationDataGen = imagesGeneratedForValidation.flow_from_directory(
        batch_size=batchSize,
        directory=validationDirectory,
        target_size=(IMG_SHAPE, IMG_SHAPE),
        class_mode='sparse')


    return trainDataGen,validationDataGen

def trainGenerator(batch_size, trainSet, aug_dict, inputSize=(256, 256), inputChannels=1):

    if batch_size > 0:
        while 1:

            iTile = 0
            nBatches = int(np.ceil(len(trainSet) / batch_size))

            for batchID in range(nBatches):

                images = np.zeros(((batch_size,) + inputSize + (inputChannels,))).astype(float)  #
                labels = np.zeros((batch_size,1)).astype(int)
                iTileInBatch = 0

                while iTileInBatch < batch_size:

                    if iTile < len(trainSet):

                        labels[iTileInBatch] = classesDict[trainSet[iTile][1]]
                        image = getImageChannels(trainSet[iTile][0])
                        image = augmentImage(image, inputSize, aug_dict,labels[iTileInBatch])



                        image = np.array(image)
                        images[iTileInBatch, :, :, :] =  image
                        images[iTileInBatch] = ((images[iTileInBatch] + 1)/2)

                        #plt.title(classes[int(labels[iTileInBatch])] )
                        #plt.imshow( images[iTileInBatch]*255 )
                        #plt.waitforbuttonpress()

                        #for i in range(len(image)):
                         #   images[iTileInBatch, :, :, i] = image[ :, :, i]

                        iTile = iTile + 1
                        iTileInBatch = iTileInBatch + 1
                    else:
                        iTile = 0

                        #images = images[0:iTileInBatch, :, :, :]
                        #break

                yield (images,labels)


def prepareDataset(datasetPath, trainFolder, valFolder):

    trainSet = []
    valSet = []


    for cl in classes:


        trainImagesPath = os.path.join(datasetPath, trainFolder, cl)
        trainSetFolder = os.scandir(trainImagesPath)

        for image in trainSetFolder:
            imagePath = image.path
            trainSet.append((imagePath,cl))

        r.shuffle(trainSet)


        valImagesPath = os.path.join(datasetPath, valFolder, cl)
        valSetXFolder = os.scandir(valImagesPath)
        for image in valSetXFolder:
            imagePath = image.path
            valSet.append((imagePath,cl))



    return trainSet,valSet




def normalizeChannel(channel):
    return (channel - 127.5) / 127.5


def getImageChannels(tile):


    channel = skimage_io.imread(tile)
    channel = skimage_transform.resize(channel, (IMG_SHAPE, IMG_SHAPE),
                                           anti_aliasing=True)
    channel = normalizeChannel(channel)
    return [channel]


def TrainModel(trainData,validationData,trainDataGen, validationDataGen):

    model = tf.keras.applications.xception.Xception(weights='imagenet')

    # Change Output layer
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
        metrics=['accuracy','sparse_categorical_accuracy'])

    validationSteps = int(np.ceil(len(validationData) / float(batchSize)))



    tensorboardCallback,modelCheckpointCallback,earlyStopCallback = DefineCallbacks()



    #Training time
    #history = model.fit(
    #    trainData,
    #    steps_per_epoch=int(np.ceil(trainData.n / float(batchSize))),
    #    epochs=epochs,
    #    validation_data=validationData,
    #    validation_steps=validationSteps,
    #    callbacks=[tensorboardCallback,modelCheckpointCallback,earlyStopCallback]
    #)

    Ntrain = len(trainData)
    stepsPerEpoch = np.ceil(Ntrain / batchSize)
    Nval = len(validationData)
    validationSteps = np.ceil(Nval / batchSize)

    history = model.fit(trainDataGen,
                        steps_per_epoch=stepsPerEpoch,
                        epochs=epochs,
                        callbacks=[modelCheckpointCallback,
                                   tensorboardCallback,earlyStopCallback],
                        validation_data=validationDataGen,
                        validation_steps=validationSteps)



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



    sns.make_confusion_matrix(confusionMatrix, figsize=(len(classes), len(classes)), cbar=False)

    print("Precision: {}\n".format(precision))
    print("Recall: {}\n".format(recall))
    print("F1: {}\n".format(f1))


def augmentImage(image, inputSize, aug_dict, label):

    defaultRotation = 0


    widthRange = (int) (image[0].shape[1]/2)
    heightRange = (int) (image[0].shape[1]/2)

    if 'width_shift_range' in aug_dict:
        cropx = r.sample(range(widthRange) ,1)[0]
    else:
        cropx = 0


    if 'height_shift_range' in aug_dict:
        cropy = r.sample(range(heightRange), 1)[0]
    else:
        cropy = 0


    if 'rotation_range' in aug_dict:
        rotation = r.sample(aug_dict['rotation_range'], 1)[0]


    if 'horizontal_flip' in aug_dict and aug_dict['horizontal_flip']:
        do_horizontal_flip = r.sample([False, True], 1)[0]
    else:
        do_horizontal_flip = False


    for i in range(len(image)):

        channel = image[i]

        #Crop
        channel = channel[cropy:cropy + inputSize[0], cropx:cropx + inputSize[1]]

        #Reshape
        channel = skimage_transform.resize(channel, (IMG_SHAPE, IMG_SHAPE),
                                           anti_aliasing=True)
        #Rotate
        channel = skimage_transform.rotate(channel, rotation)

        if do_horizontal_flip:
            channel = channel[:, ::-1]
            image[i] = channel


    image = skimage_transform.resize(image[0], (IMG_SHAPE, IMG_SHAPE),
                                       anti_aliasing=True)


    return image


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

    DownloadAndSaveImages()

    trainSet, valSet = prepareDataset(datasetPath=dataSetDirectory, trainFolder=trainingDirectory, valFolder=validationDirectory)

    trainGen = trainGenerator(batchSize, trainSet, augmentation_args, inputSize=inputSize, inputChannels=3)

    valGen = trainGenerator(batchSize, valSet, dict(), inputSize=inputSize, inputChannels=3)

    TrainModel(trainSet,valSet,trainDataGen=trainGen,validationDataGen=valGen)

    #trainDataGenerated,validationDataGenerated = DataGenerator()
    #TrainModel(trainGene,valGene,classes)
