import tensorflow as tf
from Base.createModel import createModel

IMG_HEIGHT  = 224
IMG_WIDTH   = 224
BATCH_SIZE  = 16
SEED        = 1234

strPath = "./Images/Inside"

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

bestModel = createModel(num_classes, IMG_WIDTH, IMG_HEIGHT, "MobileNetV2")

bestModel.compile(
    optimizer = opti, 
    loss = loss, 
    metrics = metr
)

bestModel.fit(
    train_ds, 
    epochs = 5, 
    validation_data = val_ds
)

bestModel.summary()