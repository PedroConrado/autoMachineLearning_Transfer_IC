import tensorflow as tf
from Base.createModel import createModel
import warnings

def transfer(
  num_classes, 
  img_width, 
  img_height, 
  train_ds, 
  val_ds,
  metr,
  opti,
  loss,
  trainingEpochs = 0,
  fineTune = False,
  fineTuneEpochs = 3,
  fineTuneLearningRate = 0.00001,
  baseModelList = ['VGG19', 'Xception', 'ResNet152V2', 'InceptionV3'],
  verbose = True
):
  
  if(not verbose):
    warnings.filterwarnings("ignore")

  modelsList = []
  [modelsList.append([createModel(num_classes, img_width, img_height, baseModel), 0]) for baseModel in baseModelList]

  for model in modelsList:
    print(model[0]._name)
    model[0].compile(
      optimizer = opti, 
      loss = loss, 
      metrics = metr
    )

  print('STARTING TRAINING')
  
  while(len(modelsList) > 1):
    print(len(modelsList), ' models remaining')
    for model in modelsList:

      print('EVALUATING: ', model[0]._name)
      
      hist = model[0].fit(
        train_ds,
        validation_data = val_ds,
        epochs = 1
      )

      model[1] = hist.history['val_accuracy']

    modelsList.sort(key = lambda x: x[1])
    for i in range(0, int(len(modelsList)/2)):
      print('erasing model ', modelsList[0][0]._name, 'with val accuracy ', modelsList[0][1])
      del modelsList[0]
  
  model = modelsList[0][0]
  print('CHOSEN MODEL: ', model._name)

  if(trainingEpochs > 0):
    model.compile(
        optimizer = opti, 
        loss = loss, 
        metrics = metr
      )
    
    hist = model.fit(
      train_ds,
      validation_data = val_ds,
      epochs = trainingEpochs
    )

  if(fineTune):
    print('STARTING FINETUNING')

    model.trainable = True

    opti = tf.keras.optimizers.RMSprop(momentum=0.1,learning_rate=fineTuneLearningRate) 

    model.compile(
      optimizer = opti, 
      loss = loss, 
      metrics = metr
    )

    model.fit(
      train_ds, 
      epochs = fineTuneEpochs, 
      validation_data = val_ds
    )

    model.summary()

  return model