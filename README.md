{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "deletable": true,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "<a href=\"https://cognitiveclass.ai\"><img src = \"https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/Logos/organization_logo/organization_logo.png\" width = 400> </a>\n",
    "\n",
    "<h1 align=center><font size = 5>Peer Review Final Assignment</font></h1>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "deletable": true,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "## Introduction\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "deletable": true,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "In this lab, you will build an image classifier using the VGG16 pre-trained model, and you will evaluate it and compare its performance to the model we built in the last module using the ResNet50 pre-trained model. Good luck!"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "deletable": true,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "## Table of Contents\n",
    "\n",
    "<div class=\"alert alert-block alert-info\" style=\"margin-top: 20px\">\n",
    "\n",
    "<font size = 3>    \n",
    "\n",
    "1. <a href=\"#item41\">Download Data \n",
    "2. <a href=\"#item42\">Part 1</a>\n",
    "3. <a href=\"#item43\">Part 2</a>  \n",
    "4. <a href=\"#item44\">Part 3</a>  \n",
    "\n",
    "</font>\n",
    "    \n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "deletable": true,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "   "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"item41\"></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Download Data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Use the <code>wget</code> command to download the data for this assignment from here: https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week4.zip"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Use the following cells to download the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "#!pip install skillsnetwork\n",
    "#!pip install keras\n",
    "#!pip install tensorflow\n",
    "#!pip install nest_asyncio"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "import skillsnetwork \n",
    "from keras.preprocessing.image import ImageDataGenerator\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "from keras.applications import VGG16\n",
    "from keras.applications.vgg16 import preprocess_input\n",
    "from keras.applications.resnet50 import preprocess_input as resnet50_pre_input"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "## get the data\n",
    "#url =  \"https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week4.zip\"\n",
    "#await skillsnetwork.prepare(url, path=path, overwrite=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "After you unzip the data, you fill find the data has already been divided into a train, validation, and test sets."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "deletable": true,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "<a id=\"item42\"></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 1"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this part, you will design a classifier using the VGG16 pre-trained model. Just like the ResNet50 model, you can import the model <code>VGG16</code> from <code>keras.applications</code>."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You will essentially build your classifier as follows:\n",
    "1. Import libraries, modules, and packages you will need. Make sure to import the *preprocess_input* function from <code>keras.applications.vgg16</code>.\n",
    "2. Use a batch size of 100 images for both training and validation.\n",
    "3. Construct an ImageDataGenerator for the training set and another one for the validation set. VGG16 was originally trained on 224 × 224 images, so make sure to address that when defining the ImageDataGenerator instances.\n",
    "4. Create a sequential model using Keras. Add VGG16 model to it and dense layer.\n",
    "5. Compile the mode using the adam optimizer and the categorical_crossentropy loss function.\n",
    "6. Fit the model on the augmented data using the ImageDataGenerators."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Use the following cells to create your classifier."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Define the constants**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#parameters for model\n",
    "num_classes = 2\n",
    "\n",
    "image_resize = 224\n",
    "\n",
    "batch_size_training = 100\n",
    "batch_size_validation = 100"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Construct the ImageDataGenerator**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#data generator\n",
    "data_generator = ImageDataGenerator(\n",
    "    preprocessing_function=preprocess_input,\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "**Get the training data using *flow_from_directory***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "scrolled": true,
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 30001 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "train_generator = data_generator.flow_from_directory(\n",
    "    'concrete_data_week4/train',\n",
    "    target_size=(image_resize, image_resize),\n",
    "    batch_size=batch_size_training,\n",
    "    class_mode='categorical')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Get the validation data using *flow_from_directory***"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 9501 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "validation_generator = data_generator.flow_from_directory(\n",
    "    'concrete_data_week4/valid',\n",
    "    target_size=(image_resize, image_resize),\n",
    "    batch_size=batch_size_training,\n",
    "    class_mode='categorical')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Build, Compile and Fit the model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "model = Sequential()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "model.add(VGG16(\n",
    "    include_top=False,\n",
    "    pooling='avg',\n",
    "    weights='imagenet',\n",
    "    ))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "model.add(Dense(num_classes, activation='softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#set the VGG16 layer not trainable\n",
    "model.layers[0].trainable = False"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " vgg16 (Functional)          (None, 512)               14714688  \n",
      "                                                                 \n",
      " dense (Dense)               (None, 2)                 1026      \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 14715714 (56.14 MB)\n",
      "Trainable params: 1026 (4.01 KB)\n",
      "Non-trainable params: 14714688 (56.13 MB)\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#compile the model\n",
    "model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "#set training steps\n",
    "steps_per_epoch_training = len(train_generator)\n",
    "steps_per_epoch_validation = len(validation_generator)\n",
    "num_epochs = 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\wanji\\AppData\\Local\\Temp\\ipykernel_5732\\2835331566.py:2: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.\n",
      "  fit_history = model.fit_generator(\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/2\n",
      "301/301 [==============================] - 7215s 24s/step - loss: 0.0839 - accuracy: 0.9754 - val_loss: 0.0302 - val_accuracy: 0.9931\n",
      "Epoch 2/2\n",
      "301/301 [==============================] - 6913s 23s/step - loss: 0.0234 - accuracy: 0.9943 - val_loss: 0.0178 - val_accuracy: 0.9952\n"
     ]
    }
   ],
   "source": [
    "#training the model\n",
    "fit_history = model.fit_generator(\n",
    "    train_generator,\n",
    "    steps_per_epoch=steps_per_epoch_training,\n",
    "    epochs=num_epochs,\n",
    "    validation_data=validation_generator,\n",
    "    validation_steps=steps_per_epoch_validation,\n",
    "    verbose=1,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\wanji\\AppData\\Roaming\\jupyterlab-desktop\\jlab_server\\lib\\site-packages\\keras\\src\\engine\\training.py:3000: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.\n",
      "  saving_api.save_model(\n"
     ]
    }
   ],
   "source": [
    "model.save('classifier_vgg_model.h5')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "   "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"item43\"></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 2"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this part, you will evaluate your deep learning models on a test data. For this part, you will need to do the following:\n",
    "\n",
    "1. Load your saved model that was built using the ResNet50 model. \n",
    "2. Construct an ImageDataGenerator for the test set. For this ImageDataGenerator instance, you only need to pass the directory of the test images, target size, and the **shuffle** parameter and set it to False.\n",
    "3. Use the **evaluate_generator** method to evaluate your models on the test data, by passing the above ImageDataGenerator as an argument. You can learn more about **evaluate_generator** [here](https://keras.io/models/sequential/).\n",
    "4. Print the performance of the classifier using the VGG16 pre-trained model.\n",
    "5. Print the performance of the classifier using the ResNet pre-trained model.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Use the following cells to evaluate your models."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "**Load classifier resnet model using ResNet50**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " resnet50 (Functional)       (None, 2048)              23587712  \n",
      "                                                                 \n",
      " dense (Dense)               (None, 2)                 4098      \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 23591810 (90.00 MB)\n",
      "Trainable params: 4098 (16.01 KB)\n",
      "Non-trainable params: 23587712 (89.98 MB)\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "from keras.saving import load_model\n",
    "ResNet50_model = load_model('classifier_resnet_model.h5')\n",
    "# summarize model.\n",
    "ResNet50_model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Construct test ImageDateGenerator**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 500 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "test_generator_vgg16 = data_generator.flow_from_directory(\n",
    "    'concrete_data_week4/test',\n",
    "    target_size=(image_resize, image_resize),\n",
    "    batch_size=batch_size_training,\n",
    "    shuffle='False')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_generator_resnet50 = ImageDataGenerator(\n",
    "    preprocessing_function=resnet50_pre_input,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 500 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "test_generator_resnet50 = data_generator_resnet50.flow_from_directory(\n",
    "    'concrete_data_week4/test',\n",
    "    target_size=(image_resize, image_resize),\n",
    "    batch_size=batch_size_training,\n",
    "    shuffle='False')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\wanji\\AppData\\Local\\Temp\\ipykernel_15404\\821735448.py:1: UserWarning: `Model.evaluate_generator` is deprecated and will be removed in a future version. Please use `Model.evaluate`, which supports generators.\n",
      "  ResNet50_score = ResNet50_model.evaluate_generator(test_generator_resnet50)\n"
     ]
    }
   ],
   "source": [
    "ResNet50_score = ResNet50_model.evaluate_generator(test_generator_resnet50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ResNet50 pre-trained model loss and accuracy are 0.001988  and  1.0\n"
     ]
    }
   ],
   "source": [
    "print(\"ResNet50 pre-trained model loss and accuracy are\", round(ResNet50_score[0],6), ' and ', ResNet50_score[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\wanji\\AppData\\Local\\Temp\\ipykernel_15404\\232058037.py:2: UserWarning: `Model.evaluate_generator` is deprecated and will be removed in a future version. Please use `Model.evaluate`, which supports generators.\n",
      "  VGG16_score = VGG16_model.evaluate_generator(test_generator_vgg16)\n"
     ]
    }
   ],
   "source": [
    "VGG16_model = load_model('classifier_vgg_model.h5')\n",
    "VGG16_score = VGG16_model.evaluate_generator(test_generator_vgg16)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "VGG16 pre-trained model loss and accuracy are 0.013523  and  0.996\n"
     ]
    }
   ],
   "source": [
    "print(\"VGG16 pre-trained model loss and accuracy are\", round(VGG16_score[0],6), ' and ', round(VGG16_score[1],6))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "   "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a id=\"item44\"></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Part 3"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this model, you will predict whether the images in the test data are images of cracked concrete or not. You will do the following:\n",
    "\n",
    "1. Use the **predict_generator** method to predict the class of the images in the test data, by passing the test data ImageDataGenerator instance defined in the previous part as an argument. You can learn more about the **predict_generator** method [here](https://keras.io/models/sequential/).\n",
    "2. Report the class predictions of the first five images in the test set. You should print something list this:\n",
    "\n",
    "<center>\n",
    "    <ul style=\"list-style-type:none\">\n",
    "        <li>Positive</li>  \n",
    "        <li>Negative</li> \n",
    "        <li>Positive</li>\n",
    "        <li>Positive</li>\n",
    "        <li>Negative</li>\n",
    "    </ul>\n",
    "</center>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Use the following cells to make your predictions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "filenames_resnet50 = test_generator_resnet50.filenames\n",
    "nb_samples_resnet50 = len(filenames_resnet50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\wanji\\AppData\\Local\\Temp\\ipykernel_15404\\1168081241.py:1: UserWarning: `Model.predict_generator` is deprecated and will be removed in a future version. Please use `Model.predict`, which supports generators.\n",
      "  ResNet50_predict = ResNet50_model.predict_generator(test_generator_resnet50, steps = nb_samples_resnet50/batch_size_training, verbose=1)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5/5 [==============================] - 33s 6s/step\n"
     ]
    }
   ],
   "source": [
    "ResNet50_predict = ResNet50_model.predict_generator(test_generator_resnet50, steps = nb_samples_resnet50/batch_size_training, verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "filenames_vgg16 = test_generator_vgg16.filenames\n",
    "nb_samples_vgg16 = len(filenames_resnet50)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\wanji\\AppData\\Local\\Temp\\ipykernel_15404\\3990815252.py:1: UserWarning: `Model.predict_generator` is deprecated and will be removed in a future version. Please use `Model.predict`, which supports generators.\n",
      "  VGG16_predict = VGG16_model.predict_generator(test_generator_vgg16, steps = nb_samples_vgg16/batch_size_training,verbose=1)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5/5 [==============================] - 90s 18s/step\n"
     ]
    }
   ],
   "source": [
    "VGG16_predict = VGG16_model.predict_generator(test_generator_vgg16, steps = nb_samples_vgg16/batch_size_training,verbose=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'negative': 0, 'positive': 1}\n",
      "ResNet50 PREDICTED CLASS     TRUE CLASS          FILENAME \n",
      "    positive                  negative           negative\\19751.jpg\n",
      "    positive                  negative           negative\\19752.jpg\n",
      "    negative                  negative           negative\\19753.jpg\n",
      "    negative                  negative           negative\\19754.jpg\n",
      "    negative                  negative           negative\\19755.jpg\n",
      "================================================================================\n",
      "{'negative': 0, 'positive': 1}\n",
      "VGG16 PREDICTED CLASS        TRUE CLASS          FILENAME \n",
      "    positive                  negative          negative\\19751.jpg\n",
      "    positive                  negative          negative\\19752.jpg\n",
      "    negative                  negative          negative\\19753.jpg\n",
      "    positive                  negative          negative\\19754.jpg\n",
      "    negative                  negative          negative\\19755.jpg\n"
     ]
    }
   ],
   "source": [
    "\n",
    "test_labels=test_generator_resnet50.labels            # list  of test labels for each image sample\n",
    "class_dict= test_generator_resnet50.class_indices     # a dictionary of the class name\n",
    "print (class_dict)                                    # print the dictionary\n",
    "new_dict={} \n",
    "for key in class_dict:                              # set key in new_dict to value in class_dict and value in new_dict to key in class_dict\n",
    "    value=class_dict[key]\n",
    "    new_dict[value] = key\n",
    "print('ResNet50 PREDICTED CLASS     TRUE CLASS          FILENAME ' ) # adjust spacing based on your class names\n",
    "for i, p in enumerate(ResNet50_predict):\n",
    "    if (i < 5):\n",
    "        pred_index = np.argmax(p)                     # get the index that has the highest probability\n",
    "        pred_class = new_dict[pred_index]             # find the predicted class based on the index\n",
    "        true_class = new_dict[test_labels[i]]         # use the test label to get the true class of the test file\n",
    "        file = filenames_resnet50[i]\n",
    "        print(f'    {pred_class}                  {true_class}           {file}')\n",
    "print(\"================================================================================\")\n",
    "\n",
    "test_labels=test_generator_vgg16.labels            # list  of test labels for each image sample\n",
    "class_dict= test_generator_vgg16.class_indices     # a dictionary of the class name\n",
    "print (class_dict)                                    # print the dictionary\n",
    "print('VGG16 PREDICTED CLASS        TRUE CLASS          FILENAME ' ) # adjust spacing based on your class names\n",
    "for i, p in enumerate(VGG16_predict):\n",
    "    if (i < 5):\n",
    "        pred_index = np.argmax(p)                     # get the index that has the highest probability\n",
    "        pred_class = new_dict[pred_index]             # find the predicted class based on the index\n",
    "        true_class = new_dict[test_labels[i]]         # use the test label to get the true class of the test file\n",
    "        file = filenames_vgg16[i]\n",
    "        print(f'    {pred_class}                  {true_class}          {file}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "  "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "deletable": true,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "### Thank you for completing this lab!\n",
    "\n",
    "This notebook was created by Alex Aklson."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "deletable": true,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "This notebook is part of a course on **Coursera** called *AI Capstone Project with Deep Learning*. If you accessed this notebook outside the course, you can take this course online by clicking [here](https://cocl.us/DL0321EN_Coursera_Week4_LAB1)."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "button": false,
    "deletable": true,
    "new_sheet": false,
    "run_control": {
     "read_only": false
    }
   },
   "source": [
    "<hr>\n",
    "\n",
    "Copyright &copy; 2020 [IBM Developer Skills Network](https://cognitiveclass.ai/?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/)."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.17"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
