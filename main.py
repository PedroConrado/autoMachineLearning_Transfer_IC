import tensorflow as tf
from Base.transferKeras import transfer

IMG_HEIGHT  = 224
IMG_WIDTH   = 224
BATCH_SIZE  = 16
SEED        = 1234

strPath = ""

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  strPath,
  validation_split = 0.2,
  subset     = "training",
  seed       = SEED,
  image_size = (IMG_HEIGHT, IMG_WIDTH),
  batch_size = BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  strPath,
  validation_split = 0.2,
  subset     = "validation",
  seed       = SEED,
  image_size = (IMG_HEIGHT, IMG_WIDTH),
  batch_size = BATCH_SIZE
)

class_names = train_ds.class_names

num_classes = len(class_names)

print(class_names)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

learningRate = 0.01
opti = tf.keras.optimizers.RMSprop(momentum=0.1, learning_rate=learningRate) 
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
metr = ['accuracy']

allModelsList = [
'Xception',
'VGG16',
'VGG19',
'ResNet50',
'ResNet50V2',
'ResNet101',
'ResNet101V2',
'ResNet152',
'ResNet152V2',
'InceptionV3',
'InceptionResNetV2',
'MobileNet',
'MobileNetV2',
'DenseNet121',
'DenseNet169',
'DenseNet201',
'NASNetMobile',
'EfficientNetB0',
'EfficientNetB1',
'EfficientNetB2',
'EfficientNetB3',
'EfficientNetB4',
'EfficientNetB5',
'EfficientNetB6',
'EfficientNetB7',
'EfficientNetV2B0',
'EfficientNetV2B1',
'EfficientNetV2B2',
'EfficientNetV2B3',
'EfficientNetV2S',
'EfficientNetV2M',
'EfficientNetV2L']

model = transfer(num_classes, IMG_WIDTH, IMG_HEIGHT, train_ds, val_ds, metr, opti, loss, trainingEpochs=5, fineTune = True, fineTuneLearningRate=0.000001, baseModelList=allModelsList, verbose=False)
