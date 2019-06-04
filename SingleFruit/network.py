from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential
from keras.models import load_model 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
from keras.preprocessing import image as kpi
import numpy  as np

def make_net(): # builds network with desired architecture and trains it using dataset saves it to file  
    
    img_width, img_height = 100, 100
    
    train_data_dir = 'ourdata/train'
    validation_data_dir = 'ourdata/test'
    nb_train_samples = 6083
    nb_validation_samples = 2031
    epochs = 8
    batch_size = 30
    
    if K.image_data_format() == 'channels_first': 
        input_shape = (3, img_width, img_height) 
    else: 
        input_shape = (img_width, img_height, 3) 
    
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='sigmoid'))
    
    model.compile(loss ='binary_crossentropy', 
                  optimizer ='rmsprop', 
                  metrics =['accuracy']) 
    
    train_datagen = ImageDataGenerator( 
        rescale = 1. / 255, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True) 
    
    test_datagen = ImageDataGenerator(rescale = 1. / 255) 
    
    train_generator = train_datagen.flow_from_directory(train_data_dir, 
                                                        target_size =(img_width, img_height), 
                                                        batch_size = batch_size, class_mode ='categorical') 
    
    validation_generator = test_datagen.flow_from_directory(validation_data_dir, 
                                                            target_size =(img_width, img_height), 
                                                            batch_size = batch_size, class_mode ='categorical') 
    
    model.fit_generator(train_generator, 
                        steps_per_epoch = nb_train_samples // batch_size,
                        epochs = epochs, validation_data = validation_generator, 
                        validation_steps = nb_validation_samples // batch_size) 
    
    model.save_weights('testing2.h5') 
    return model

def open_net(netfile): # builds network with desired achitecture and loads weights from file
    img_width, img_height = 100, 100
    
    train_data_dir = 'ourdata/train'
    validation_data_dir = 'ourdata/test'
    
    if K.image_data_format() == 'channels_first': 
        input_shape = (3, img_width, img_height) 
    else: 
        input_shape = (img_width, img_height, 3) 
    
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='sigmoid'))
    
    model.compile(loss ='binary_crossentropy', 
                  optimizer ='rmsprop', 
                  metrics =['accuracy']) 
    
    train_datagen = ImageDataGenerator( 
        rescale = 1. / 255, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True) 
    
    test_datagen = ImageDataGenerator(rescale = 1. / 255) 
    
    
    model.load_weights(netfile)
    return model


def classify(model, filename):       # classifies image from filename using loaded model, prints results
    img_width, img_height = 100, 100
    
    train_data_dir = 'ourdata/train'
    validation_data_dir = 'ourdata/test'
    batch_size = 30
    
    train_datagen = ImageDataGenerator( 
        rescale = 1. / 255, 
        shear_range = 0.2, 
        zoom_range = 0.2, 
        horizontal_flip = True)     
    train_generator = train_datagen.flow_from_directory(train_data_dir, 
                                                        target_size =(img_width, img_height), 
                                                        batch_size = batch_size, class_mode ='categorical') 
    
    
    img = kpi.load_img(filename, target_size=(100, 100))
    img = kpi.array_to_img(img)
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    #print(train_generator.class_indices)
    #print(result)
    #print(result[0][train_generator.class_indices[tag]])
    maxr = 0.5
    besttag= "Other"
    for tag in list(train_generator.class_indices):
        string = tag
        string = string + " = " + str(result[0][train_generator.class_indices[tag]] * 100) + " %"
        print(string)
        if (result[0][train_generator.class_indices[tag]] >= maxr):
            maxr = result[0][train_generator.class_indices[tag]]
            besttag = tag
    return besttag, maxr

#make_net()