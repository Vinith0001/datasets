20.Hate speech detection
# ============================================================
# 🧠 Hate Speech & Offensive Language Detection using TensorFlow
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 1️⃣  Sample Dataset (You can replace this with a real CSV)
# ============================================================

texts = [
    "I hate you so much!",
    "You are such a loser.",
    "Love and peace for everyone.",
    "What a wonderful day!",
    "You are disgusting and awful!",
    "I really appreciate your help.",
    "That was a stupid thing to do.",
    "You are kind and amazing!",
    "Go die somewhere.",
    "Have a great weekend ahead!"
]

# Labels: 1 = Hate/Offensive, 0 = Normal
labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]

# ============================================================
# 2️⃣  Tokenization and Padding
# ============================================================

vocab_size = 5000
max_len = 20

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=max_len, padding='post')
y = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ============================================================
# 3️⃣  Build the Deep Learning Model
# ============================================================

model = Sequential([
    Embedding(vocab_size, 64, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# ============================================================
# 4️⃣  Compile and Train the Model
# ============================================================

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\n🧩 Model Summary:")
model.summary()

history = model.fit(X_train, y_train, epochs=10, batch_size=2, validation_split=0.2, verbose=1)

# ============================================================
# 5️⃣  Evaluate the Model
# ============================================================

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Model Evaluation:\nLoss: {loss:.4f}\nAccuracy: {accuracy:.4f}")

# ============================================================
# 6️⃣  Sample Predictions
# ============================================================

sample_texts = [
    "I love your positive attitude!",
    "You are the worst person ever.",
    "I hope you fail miserably.",
    "Have a blessed and happy life!"
]

# Tokenize and pad
sample_seq = tokenizer.texts_to_sequences(sample_texts)
sample_pad = pad_sequences(sample_seq, maxlen=max_len, padding='post')

# Predict
predictions = model.predict(sample_pad)
pred_classes = (predictions > 0.5).astype("int32")

print("\n🔍 Sample Predictions:")
for text, pred in zip(sample_texts, pred_classes):
    label = "🚫 Hate Speech" if pred == 1 else "💬 Normal"
    print(f"Text: {text}\n→ Prediction: {label}\n")

# ============================================================
# 7️⃣  Model Evaluation Report
# ============================================================

y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Hate Speech"]))

19. fake news prediction
# ============================================================
# 🧠 Fake News Detection using TensorFlow Deep Learning Model
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ============================================================
# 1️⃣ Dataset Preparation
# (Sample dataset for demo — replace with real dataset later)
# ============================================================

texts = [
    "Breaking: Scientists discover cure for cancer after 50 years!",
    "Celebrity says the earth is flat during an interview.",
    "Government announces new AI policy for 2025.",
    "Fake news spreads about alien invasion in New York.",
    "NASA confirms water found on Mars surface.",
    "Click here to win a free iPhone instantly!",
    "Economy expected to grow 5% next quarter, experts say.",
    "Hoax: Man claims to have traveled through time.",
    "COVID-19 vaccine proves highly effective in trials.",
    "Conspiracy theory suggests moon landing was fake."
]

# Labels: 1 = Fake, 0 = Real
labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

# ============================================================
# 2️⃣ Tokenization and Padding
# ============================================================

vocab_size = 5000
max_len = 30

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=max_len, padding='post')
y = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ============================================================
# 3️⃣ Model Building
# ============================================================

model = Sequential([
    Embedding(vocab_size, 64, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

print("\n🧩 Model Summary:")
model.summary()

# ============================================================
# 4️⃣ Model Compilation & Training
# ============================================================

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, batch_size=2, validation_split=0.2, verbose=1)

# ============================================================
# 5️⃣ Evaluation
# ============================================================

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Model Evaluation:\nLoss: {loss:.4f}\nAccuracy: {accuracy:.4f}")

# ============================================================
# 6️⃣ Sample Input & Prediction
# ============================================================

sample_news = [
    "Breaking: NASA announces mission to Jupiter for 2030!",
    "Scientists confirm that chocolate causes instant weight loss.",
    "Local elections see record voter turnout this year.",
    "Aliens spotted in Los Angeles according to viral video."
]

sample_seq = tokenizer.texts_to_sequences(sample_news)
sample_pad = pad_sequences(sample_seq, maxlen=max_len, padding='post')

predictions = model.predict(sample_pad)
pred_classes = (predictions > 0.5).astype("int32")

print("\n🔍 Sample Predictions:")
for text, pred in zip(sample_news, pred_classes):
    label = "🧠 Real News" if pred == 0 else "⚠️ Fake News"
    print(f"Text: {text}\n→ Prediction: {label}\n")

# ============================================================
# 7️⃣ Final Evaluation Report
# ============================================================

y_pred = (model.predict(X_test) > 0.5).astype("int32")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

18/ Gan medical augumentation 

# gan_medical_augmentation.py
import os
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

# ============================================================
# 1️⃣ DATASET PREPARATION
# ============================================================
# Settings
DATA_DIR = "data/medical/train"   # put your medical images here (jpg/png). All images in this folder or subfolders.
IMG_SIZE = 128                    # output image size (square)
BATCH_SIZE = 32
BUFFER_SIZE = 1000
CHANNELS = 1                      # use 1 for grayscale, 3 for RGB
AUTOTUNE = tf.data.AUTOTUNE

def load_and_preprocess(path):
    # Load image file, convert to grayscale if needed, resize and normalize to [-1,1]
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = (img - 0.5) * 2.0  # scale to [-1, 1]
    img.set_shape([IMG_SIZE, IMG_SIZE, CHANNELS])
    return img

# Build tf.data.Dataset
image_files = []
for ext in ("*.png","*.jpg","*.jpeg","*.bmp"):
    image_files.extend(glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True))
image_files = sorted(image_files)
print(f"Found {len(image_files)} images")

if len(image_files) == 0:
    raise SystemExit("No images found. Put medical images in data/medical/train/")

ds = tf.data.Dataset.from_tensor_slices(image_files)
ds = ds.shuffle(len(image_files))
ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
ds = ds.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

# ============================================================
# 2️⃣ MODEL BUILDING (Generator & Discriminator)
# ============================================================
# Latent dim (noise vector)
LATENT_DIM = 100

# Generator: noise -> IMG_SIZE x IMG_SIZE x CHANNELS
def build_generator(latent_dim=LATENT_DIM, channels=CHANNELS):
    model = tf.keras.Sequential(name="Generator")
    model.add(tf.keras.layers.Input(shape=(latent_dim,)))
    # project and reshape
    nodes = 8 * 8 * 256
    model.add(tf.keras.layers.Dense(nodes, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Reshape((8, 8, 256)))  # (8,8,256)

    # Upsample to 16x16
    model.add(tf.keras.layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Upsample to 32x32
    model.add(tf.keras.layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Upsample to 64x64
    model.add(tf.keras.layers.Conv2DTranspose(32, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Upsample to 128x128
    model.add(tf.keras.layers.Conv2DTranspose(16, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    # Final conv -> channels with tanh
    model.add(tf.keras.layers.Conv2D(channels, (7,7), padding='same', activation='tanh'))
    return model

# Discriminator: IMG_SIZE x IMG_SIZE x CHANNELS -> real/fake
def build_discriminator(channels=CHANNELS):
    model = tf.keras.Sequential(name="Discriminator")
    model.add(tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, channels)))
    model.add(tf.keras.layers.Conv2D(32, (5,5), strides=(2,2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(64, (5,5), strides=(2,2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))  # logits
    return model

generator = build_generator()
discriminator = build_discriminator()

# Show summaries
print("\nGenerator summary:")
generator.summary()
print("\nDiscriminator summary:")
discriminator.summary()

# ============================================================
# 3️⃣ LOSS, OPTIMIZERS & CHECKPOINTS
# ============================================================
# Use logits with from_logits=True
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_logits, fake_logits):
    real_loss = bce(tf.ones_like(real_logits), real_logits)
    fake_loss = bce(tf.zeros_like(fake_logits), fake_logits)
    return real_loss + fake_loss

def generator_loss(fake_logits):
    return bce(tf.ones_like(fake_logits), fake_logits)

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

# Checkpointing
CHECKPOINT_DIR = "./gan_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator,
                                 gen_opt=generator_optimizer,
                                 disc_opt=discriminator_optimizer)
manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=3)

# ============================================================
# 4️⃣ TRAINING LOOP
# ============================================================
EPOCHS = 100      # increase when training for real use (100+)
NUM_EXAMPLES_TO_GENERATE = 16
seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, LATENT_DIM])

# Directory to save generated images
SAMPLES_DIR = "generated_samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)

@tf.function
def train_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)

        real_logits = discriminator(real_images, training=True)
        fake_logits = discriminator(fake_images, training=True)

        gen_loss = generator_loss(fake_logits)
        disc_loss = discriminator_loss(real_logits, fake_logits)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input, folder=SAMPLES_DIR):
    predictions = model(test_input, training=False)
    # rescale from [-1,1] to [0,1]
    preds = (predictions + 1.0) / 2.0
    fig = plt.figure(figsize=(4,4))
    for i in range(preds.shape[0]):
        plt.subplot(4,4,i+1)
        if CHANNELS == 1:
            plt.imshow(preds[i,:,:,0], cmap='gray')
        else:
            plt.imshow(preds[i])
        plt.axis('off')
    plt.suptitle(f"Epoch {epoch}")
    out_path = os.path.join(folder, f"epoch_{epoch:03d}.png")
    plt.savefig(out_path)
    plt.close(fig)

# Training loop
def train(dataset, epochs):
    step = 0
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        prog = tqdm(dataset, total=int(len(image_files)//BATCH_SIZE))
        for image_batch in prog:
            gen_loss, disc_loss = train_step(image_batch)
            step += 1
            prog.set_postfix({'g_loss': float(gen_loss.numpy()), 'd_loss': float(disc_loss.numpy())})
        # Save generated sample images and checkpoint after each epoch
        generate_and_save_images(generator, epoch, seed)
        manager.save()
    # final sample
    generate_and_save_images(generator, 'final', seed)

# ============================================================
# 5️⃣ RUN TRAINING (Start)
# ============================================================
if __name__ == "__main__":
    train(ds, EPOCHS)

    # ============================================================
    # 6️⃣ SAMPLE INPUT & PRODUCE AUGMENTED IMAGES (PREDICTION)
    # ============================================================
    # Generate N images to use for augmentation
    N_GEN = 500  # how many synthetic images to create
    out_aug_dir = "augmented_images"
    os.makedirs(out_aug_dir, exist_ok=True)

    steps = (N_GEN + BATCH_SIZE - 1) // BATCH_SIZE
    gen_count = 0
    for _ in range(steps):
        z = tf.random.normal([BATCH_SIZE, LATENT_DIM])
        fake = generator(z, training=False)   # shape [BATCH_SIZE, IMG_SIZE, IMG_SIZE, channels]
        fake = (fake + 1.0) / 2.0              # [-1,1] -> [0,1]
        fake = tf.clip_by_value(fake, 0.0, 1.0)
        fake_np = (fake.numpy() * 255).astype(np.uint8)
        for i in range(BATCH_SIZE):
            if gen_count >= N_GEN: break
            filename = os.path.join(out_aug_dir, f"gen_{gen_count:05d}.png")
            if CHANNELS == 1:
                Image.fromarray(fake_np[i,:,:,0]).save(filename)
            else:
                Image.fromarray(fake_np[i]).save(filename)
            gen_count += 1
        if gen_count >= N_GEN:
            break
    print(f"Saved {gen_count} generated images to {out_aug_dir}")

    # ============================================================
    # 7️⃣ SIMPLE QUALITATIVE EVALUATION / VISUAL CHECK
    # ============================================================
    # Show some generated images
    sample_paths = sorted(glob.glob(os.path.join(out_aug_dir, "*.png")))[:16]
    n = len(sample_paths)
    cols = 4
    rows = (n + cols - 1)//cols
    plt.figure(figsize=(cols*2, rows*2))
    for i, p in enumerate(sample_paths):
        img = Image.open(p).convert('L' if CHANNELS==1 else 'RGB')
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap='gray' if CHANNELS==1 else None)
        plt.axis('off')
    plt.suptitle("Sample generated images (qualitative check)")
    plt.show()

    # ============================================================
    # 8️⃣ FINAL REPORT / NEXT STEPS
    # ============================================================
    print("GAN training finished. Generated images are saved in:", out_aug_dir)
    print("Use these generated images to augment your real dataset. E.g., mix them into training folder and retrain your classifier.")
17. animal image augmentation 
# ============================================================
# GAN for Animal Image Augmentation (Cats vs Dogs Dataset)
# ============================================================

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# ============================================================
# 1️⃣ LOAD DATASET (TensorFlow’s built-in cats vs dogs)
# ============================================================
url = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=url, extract=True)
data_dir = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered/train')

# Load and preprocess images
IMG_SIZE = 64
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode=None
)

# Normalize images to [-1, 1]
def normalize_img(img):
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1
    return img

train_images = train_ds.map(lambda x: normalize_img(x)).unbatch().batch(BATCH_SIZE)

# ============================================================
# 2️⃣ DEFINE GENERATOR NETWORK
# ============================================================
def make_generator_model():
    model = tf.keras.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

generator = make_generator_model()
generator.summary()

# ============================================================
# 3️⃣ DEFINE DISCRIMINATOR NETWORK
# ============================================================
def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[IMG_SIZE, IMG_SIZE, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

discriminator = make_discriminator_model()
discriminator.summary()

# ============================================================
# 4️⃣ DEFINE LOSSES & OPTIMIZERS
# ============================================================
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# ============================================================
# 5️⃣ TRAINING SETUP
# ============================================================
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = (predictions + 1) / 2.0
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i])
        plt.axis('off')
    plt.suptitle(f'Generated Samples - Epoch {epoch}')
    plt.show()

def train(dataset, epochs):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch)
        print(f"Generator Loss: {g_loss.numpy():.4f}, Discriminator Loss: {d_loss.numpy():.4f}")
        generate_and_save_images(generator, epoch + 1, seed)

train(train_images, EPOCHS)

# ============================================================
# 6️⃣ GENERATE AUGMENTED IMAGES (PREDICTION)
# ============================================================
noise = tf.random.normal([16, noise_dim])
generated_images = generator(noise, training=False)
generated_images = (generated_images + 1) / 2.0  # Rescale to [0, 1]

plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated_images[i])
    plt.axis('off')
plt.suptitle("Generated Animal Images (Augmented Data)")
plt.show()

# ============================================================
# 7️⃣ EVALUATION OF MODEL PERFORMANCE
# ============================================================
real_batch = next(iter(train_images))
real_preds = discriminator(real_batch, training=False)
fake_preds = discriminator(generator(tf.random.normal([BATCH_SIZE, noise_dim]), training=False), training=False)

print("\n--- Model Evaluation ---")
print("Average Real Image Score:", tf.reduce_mean(real_preds).numpy())
print("Average Fake Image Score:", tf.reduce_mean(fake_preds).numpy())
print("✅ The discriminator should give higher scores for real images than fake ones.")

16.English → French Translation using Encoder–Decoder

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# 1️⃣ Load dataset
data, info = tfds.load('ted_hrlr_translate/en_to_fr', with_info=True, as_supervised=True)
train_examples, val_examples = data['train'], data['validation']

# 2️⃣ Prepare subword tokenizer
tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for en, fr in train_examples), target_vocab_size=2**13)
tokenizer_fr = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    (fr.numpy() for en, fr in train_examples), target_vocab_size=2**13)

# Helper encode function
def encode(en, fr):
    en = [tokenizer_en.vocab_size] + tokenizer_en.encode(en.numpy()) + [tokenizer_en.vocab_size + 1]
    fr = [tokenizer_fr.vocab_size] + tokenizer_fr.encode(fr.numpy()) + [tokenizer_fr.vocab_size + 1]
    return en, fr

def tf_encode(en, fr):
    return tf.py_function(encode, [en, fr], [tf.int64, tf.int64])

# Preprocess dataset
BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_LEN = 40

train_data = train_examples.map(tf_encode)
train_data = train_data.filter(lambda x, y: tf.logical_and(tf.size(x) <= MAX_LEN, tf.size(y) <= MAX_LEN))
train_data = train_data.cache().shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))

val_data = val_examples.map(tf_encode)
val_data = val_data.padded_batch(BATCH_SIZE, padded_shapes=([None], [None]))

# 3️⃣ Build Encoder
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super().__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=True)
    def call(self, x):
        x = self.embedding(x)
        output, state = self.gru(x)
        return output, state

# 4️⃣ Build Decoder
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return self.fc(output), state

# Initialize models
embedding_dim = 256
units = 512
encoder = Encoder(tokenizer_en.vocab_size + 2, embedding_dim, units)
decoder = Decoder(tokenizer_fr.vocab_size + 2, embedding_dim, units)

# 5️⃣ Define optimizer and loss
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

@tf.function
def train_step(inp, targ):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([tokenizer_fr.vocab_size] * BATCH_SIZE, 1)
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden = decoder(dec_input, dec_hidden)
            loss += loss_fn(targ[:, t], predictions[:, -1, :])
            dec_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = tf.reduce_mean(loss)
    variables = encoder.trainable_variables + decoder.trainable_variables
    optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
    return batch_loss

# 6️⃣ Train model briefly (for demonstration)
for epoch in range(1):
    total_loss = 0
    for (batch, (inp, targ)) in enumerate(train_data.take(50)):  # small subset
        batch_loss = train_step(inp, targ)
        total_loss += batch_loss
    print(f"Epoch {epoch+1}, Loss: {total_loss/50:.4f}")

# 7️⃣ Translation function
def translate(sentence):
    inputs = [tokenizer_en.vocab_size] + tokenizer_en.encode(sentence) + [tokenizer_en.vocab_size + 1]
    inputs = tf.expand_dims(inputs, 0)
    enc_out, enc_hidden = encoder(inputs)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([tokenizer_fr.vocab_size], 0)
    result = []
    for t in range(40):
        predictions, dec_hidden = decoder(dec_input, dec_hidden)
        predicted_id = tf.argmax(predictions[0, -1, :]).numpy()
        if predicted_id == tokenizer_fr.vocab_size + 1:
            break
        result.append(predicted_id)
        dec_input = tf.expand_dims([predicted_id], 0)
    return tokenizer_fr.decode(result)

# 8️⃣ Sample prediction
sample_sentences = [
    "How are you?",
    "I love deep learning.",
    "Where is the library?",
    "You are my friend."
]

print("\n🔍 Sample Translations:")
for s in sample_sentences:
    print(f"English: {s}")
    print(f"French : {translate(s)}\n")

# 9️⃣ Evaluation (simple BLEU-like metric on small test)
from nltk.translate.bleu_score import sentence_bleu
scores = []
for en, fr in val_examples.take(5):
    pred = translate(en.numpy().decode())
    ref = fr.numpy().decode()
    score = sentence_bleu([[ref.split()]], pred.split())
    scores.append(score)
print(f"Average BLEU score on small validation set: {np.mean(scores):.3f}")


14. seq-to-seq transalation
"""
English -> POS tags (Sequence-to-Sequence) using TensorFlow (Keras)
Structured steps:
1) Dataset preparation (NLTK Treebank, universal tagset)
2) Build encoder-decoder seq2seq model
3) Train (teacher forcing)
4) Predict (greedy decoding), print sample inputs and outputs
5) Evaluate (token-level accuracy and classification report)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk

# -------------------------
# 0) Ensure NLTK resources
# -------------------------
nltk_data_needed = ["treebank", "universal_tagset"]
for res in nltk_data_needed:
    try:
        nltk.data.find(f"corpora/{res}")
    except LookupError:
        nltk.download(res)

# -------------------------
# 1) DATASET PREPARATION
# -------------------------
# We'll use NLTK's treebank tagged sentences and convert tags to universal tagset
tagged_sents = nltk.corpus.treebank.tagged_sents(tagset='universal')  # list of [ (word, tag), ... ]

# For demo speed, limit number of sentences; increase for better training
NUM_SENTS = 5000            # increase to use more data; treebank has ~5000+ sentences
sentences = []
pos_tags = []
for sent in tagged_sents[:NUM_SENTS]:
    words = [w for w, t in sent]
    tags  = [t for w, t in sent]
    sentences.append([w.lower() for w in words])
    pos_tags.append(tags)

# Build word tokenizer (simple index mapping)
# Reserve:
# 0 -> PAD
# 1 -> OOV
word2idx = {"<PAD>":0, "<OOV>":1}
for sent in sentences:
    for w in sent:
        if w not in word2idx:
            word2idx[w] = len(word2idx)
idx2word = {i:w for w,i in word2idx.items()}
vocab_size = len(word2idx)

# Build tag to index mapping (small set of universal POS tags)
unique_tags = sorted(list({t for seq in pos_tags for t in seq}))
# Reserve 0 for PAD, 1 for START token (decoder input), 2 for OOV-tag (unlikely)
tag2idx = {"<PAD>":0, "<START>":1}
for t in unique_tags:
    tag2idx[t] = len(tag2idx)
idx2tag = {i:t for t,i in tag2idx.items()}
num_tags = len(tag2idx)

print("Dataset size:", len(sentences))
print("Vocab size:", vocab_size, "Num tags:", num_tags)
print("Tags:", unique_tags)

# Convert sentences/tags to sequences of indices
sent_seq = [[word2idx.get(w, word2idx["<OOV>"]) for w in s] for s in sentences]
tag_seq  = [[tag2idx[t] for t in seq] for seq in pos_tags]

# Add START token to decoder inputs (for training)
decoder_input_seq = [[tag2idx["<START>"]] + ts for ts in tag_seq]
decoder_target_seq = [ts + [tag2idx["<PAD>"]] for ts in tag_seq]  # shifted teacher-forcing target (pad at end)

# Pad sequences to same length
MAX_LEN = max(len(s) for s in sent_seq)
print("Max sentence length:", MAX_LEN)

encoder_input = pad_sequences(sent_seq, maxlen=MAX_LEN, padding='post', truncating='post')
decoder_input = pad_sequences(decoder_input_seq, maxlen=MAX_LEN, padding='post', truncating='post')
decoder_target = pad_sequences(decoder_target_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# Split
X_enc_train, X_enc_test, X_dec_train, X_dec_test, y_train, y_test = train_test_split(
    encoder_input, decoder_input, decoder_target, test_size=0.2, random_state=42
)

print("Shapes -> encoder_input:", encoder_input.shape, "decoder_input:", decoder_input.shape)

# -------------------------
# 2) MODEL BUILDING (Encoder-Decoder)
# -------------------------
EMBED_DIM = 128
ENC_UNITS = 256
DEC_UNITS = 256

# Encoder
enc_inputs = Input(shape=(MAX_LEN,), name='enc_inputs')
enc_emb = layers.Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, mask_zero=True, name='enc_emb')(enc_inputs)
enc_lstm = layers.LSTM(ENC_UNITS, return_state=True, return_sequences=True, name='enc_lstm')
enc_outputs, enc_h, enc_c = enc_lstm(enc_emb)

# Decoder (training uses whole target shifted sequences)
dec_inputs = Input(shape=(MAX_LEN,), name='dec_inputs')  # token ids of tags (teacher forcing)
dec_emb_layer = layers.Embedding(input_dim=num_tags, output_dim=EMBED_DIM, mask_zero=True, name='dec_emb')
dec_emb = dec_emb_layer(dec_inputs)
dec_lstm = layers.LSTM(DEC_UNITS, return_sequences=True, return_state=True, name='dec_lstm')
# Initialize decoder state with encoder final state (project if dims differ)
# Here ENC_UNITS == DEC_UNITS so we can reuse directly; otherwise add Dense projection
dec_outputs, _, _ = dec_lstm(dec_emb, initial_state=[enc_h, enc_c])
# Final TimeDistributed dense -> predict tag logits at each timestep
dec_dense = layers.TimeDistributed(layers.Dense(num_tags, activation='softmax'), name='dec_dense')
dec_outputs = dec_dense(dec_outputs)

# Seq2Seq model (training)
model = Model([enc_inputs, dec_inputs], dec_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

# -------------------------
# 3) TRAINING
# -------------------------
# Note: decoder_target needs shape (batch, timesteps, 1) for sparse_categorical_crossentropy;
# Keras handles that if we pass decoder_target with shape (batch, timesteps).
EPOCHS = 5   # increase for better performance
BATCH = 64

history = model.fit(
    [X_enc_train, X_dec_train],
    y_train,
    batch_size=BATCH,
    epochs=EPOCHS,
    validation_split=0.1,
    verbose=2
)

# -------------------------
# 4) INFERENCE MODELS (encoder_model + decoder_step_model)
# -------------------------
# Encoder model: returns encoder states (h,c)
encoder_model_inf = Model(enc_inputs, [enc_h, enc_c])

# Decoder step model: given previous tag token and previous states -> next probabilities + new states
dec_state_h = Input(shape=(DEC_UNITS,), name='dec_state_h')
dec_state_c = Input(shape=(DEC_UNITS,), name='dec_state_c')
dec_emb2 = dec_emb_layer(dec_inputs)  # reuse embedding layer
dec_outputs2, state_h2, state_c2 = dec_lstm(dec_emb2, initial_state=[dec_state_h, dec_state_c])
dec_outputs2 = dec_dense(dec_outputs2)
# Model expects dec_inputs shape (1, t) but we'll run with t=1 during autoreg
decoder_model_inf = Model([dec_inputs, dec_state_h, dec_state_c], [dec_outputs2, state_h2, state_c2])

# -------------------------
# 5) PREDICTION FUNCTION (greedy decoding)
# -------------------------
def decode_sequence_greedy(input_seq):
    """
    input_seq: 1D array shape (MAX_LEN,) of word indices (already padded)
    Returns: list of predicted tag strings (length = original non-pad length)
    """
    # get encoder states
    input_seq = input_seq.reshape(1, -1)
    enc_h_inf, enc_c_inf = encoder_model_inf.predict(input_seq, verbose=0)
    # start token as decoder input
    cur_token = np.array([[tag2idx["<START>"]]], dtype=np.int32)  # shape (1,1)
    decoded_tags = []
    state_h, state_c = enc_h_inf, enc_c_inf

    for t in range(MAX_LEN):
        # run one step
        preds, state_h, state_c = decoder_model_inf.predict([cur_token, state_h, state_c], verbose=0)
        # preds shape: (1, 1, num_tags) if dec_inputs len 1
        pred_id = int(np.argmax(preds[0, -1, :]))
        if pred_id == tag2idx["<PAD>"]:  # reached padding or end
            break
        decoded_tags.append(idx2tag.get(pred_id, "<UNK>"))
        # next input token is predicted tag id
        cur_token = np.array([[pred_id]], dtype=np.int32)
    return decoded_tags

# -------------------------
# 6) SAMPLE PREDICTIONS (print inputs and outputs)
# -------------------------
def idxs_to_words(idxs):
    return [idx2word.get(i, "<OOV>") for i in idxs]

print("\n=== Sample predictions on test set ===")
num_samples = 8
for i in range(num_samples):
    inp = X_enc_test[i]
    nonpad_len = (inp != 0).sum()
    words = idxs_to_words(inp[:nonpad_len])
    pred_tags = decode_sequence_greedy(inp)
    print(f"\nSentence {i+1}:")
    print("Words :"," ".join(words))
    print("Pred  :", " ".join(pred_tags))

# -------------------------
# 7) EVALUATION (token-level)
# -------------------------
# Build flattened lists of true vs predicted tags (ignore PAD tokens)
y_true_flat = []
y_pred_flat = []
for i in range(len(X_enc_test)):
    inp = X_enc_test[i]
    nonpad_len = (inp != 0).sum()
    true_tags_idx = y_test[i][:nonpad_len]  # targets aligned
    pred_tags = decode_sequence_greedy(inp)
    # If predicted shorter, pad with PAD; if longer, cut
    pred_idx_seq = [tag2idx.get(t, tag2idx["<PAD>"]) for t in pred_tags]
    # equalize lengths
    if len(pred_idx_seq) < nonpad_len:
        pred_idx_seq += [tag2idx["<PAD>"]] * (nonpad_len - len(pred_idx_seq))
    else:
        pred_idx_seq = pred_idx_seq[:nonpad_len]
    y_true_flat.extend([idx2tag[idx] for idx in true_tags_idx])
    y_pred_flat.extend([idx2tag.get(idx, "<UNK>") for idx in pred_idx_seq])

print("\nToken-level classification report (test set):")
print(classification_report(y_true_flat, y_pred_flat, zero_division=0))

# -------------------------
# 8) Final: print overall token-level accuracy
# -------------------------
acc = np.mean([1 if t==p else 0 for t,p in zip(y_true_flat, y_pred_flat)])
print(f"\nToken-level accuracy (approx): {acc:.4f}")

# =========================
# Notes / Next steps:
# - Increase NUM_SENTS and EPOCHS for better performance
# - You can replace encoder LSTM with Bidirectional and project states to decoder dim
# - Use attention mechanism for better alignment (optional)
# =========================




13.13. Implement parts of speech tagging using Sequence to Sequence architecture.
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Sample dataset (English sentences with POS tags)
sentences = [
    "I love dogs",
    "She eats apples",
    "They play football",
    "He runs fast",
    "Birds fly high"
]

pos_tags = [
    "PRON VERB NOUN",
    "PRON VERB NOUN",
    "PRON VERB NOUN",
    "PRON VERB ADV",
    "NOUN VERB ADV"
]

# Tokenize words (input)
tokenizer_input = Tokenizer()
tokenizer_input.fit_on_texts(sentences)
X = tokenizer_input.texts_to_sequences(sentences)
X = pad_sequences(X, padding='post')

# Tokenize POS tags (output)
tokenizer_output = Tokenizer()
tokenizer_output.fit_on_texts(pos_tags)
y = tokenizer_output.texts_to_sequences(pos_tags)
y = pad_sequences(y, padding='post')

# Vocabulary sizes
input_vocab = len(tokenizer_input.word_index) + 1
output_vocab = len(tokenizer_output.word_index) + 1
max_seq_len = X.shape[1]

# Build Seq2Seq Model
embedding_dim = 64
latent_dim = 128

# Encoder
encoder_inputs = Input(shape=(max_seq_len,))
x = Embedding(input_vocab, embedding_dim, mask_zero=True)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(x)

# Decoder
decoder_inputs = Input(shape=(max_seq_len,))
dec_emb = Embedding(output_vocab, embedding_dim, mask_zero=True)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True)
decoder_outputs = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
decoder_dense = Dense(output_vocab, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Full Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile & Train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([X, y], np.expand_dims(y, -1), epochs=300, verbose=0)

# Evaluate
loss, acc = model.evaluate([X, y], np.expand_dims(y, -1), verbose=0)
print(f"\nModel Accuracy: {acc:.4f}")

# --- Predict POS tags for sample sentences ---
reverse_word_index = {v: k for k, v in tokenizer_output.word_index.items()}

def predict_pos(sentence):
    seq = tokenizer_input.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=max_seq_len, padding='post')
    pred = model.predict([seq, np.zeros_like(seq)], verbose=0)
    pred_tags = np.argmax(pred[0], axis=-1)
    tags = [reverse_word_index.get(i, '') for i in pred_tags if i != 0]
    return tags

# --- Display sample predictions ---
samples = ["I play football", "Birds fly", "She runs fast"]
print("\nSample Predictions:")
for s in samples:
    print(f"Sentence: {s}")
    print(f"Predicted POS Tags: {predict_pos(s)}\n")





12.12. Build an LSTM-based sentiment analysis model for Twitter data. Illustrate how LSTMs capture sequential information and address the challenges of analyzing tweets

      # Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
import numpy as np

# ----------------------------
# Step 1: Load and preprocess data
# ----------------------------
# Using IMDB dataset (acts as Twitter-style sentiment data)
# num_words = vocabulary size
num_words = 10000
maxlen = 100

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Padding sequences to make all sequences of same length
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# ----------------------------
# Step 2: Build the LSTM model
# ----------------------------
model = Sequential([
    Embedding(num_words, 128, input_length=maxlen),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (Positive/Negative)
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ----------------------------
# Step 3: Train the model
# ----------------------------
history = model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.2, verbose=1)

# ----------------------------
# Step 4: Evaluate the model
# ----------------------------
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\n🧾 Model Evaluation:")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# ----------------------------
# Step 5: Print sample input & output
# ----------------------------
# Decode sample reviews back to words
word_index = imdb.get_word_index()
reverse_word_index = {value + 3: key for key, value in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"
reverse_word_index[3] = "<UNUSED>"

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# Pick a few sample test reviews
for i in range(3):
    sample_input = x_test[i]
    print(f"\n📜 Sample Review {i+1}:")
    print(decode_review(sample_input))
    
    prediction = model.predict(np.expand_dims(sample_input, axis=0), verbose=0)
    sentiment = "😊 Positive" if prediction[0][0] > 0.5 else "😡 Negative"
    print(f"🔮 Predicted Sentiment: {sentiment}")
    print(f"Actual Label: {'Positive' if y_test[i]==1 else 'Negative'}")

# ----------------------------
# Step 6: Display Sequential Nature Explanation
# ----------------------------
print("\n🧩 Explanation:")
print("LSTMs capture the sequential nature of text, meaning that the order of words matters.")
print("Unlike traditional models, LSTMs remember context across words — for instance,")
print("‘not good’ is recognized as negative because LSTM understands ‘not’ modifies ‘good’.")


11.. Perform sentiment analysis on movie reviews using Long Short-Term Memory (LSTM) networks with the IMDB dataset
     # Import necessary libraries
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import numpy as np

# ----------------------------
# Step 1: Load and preprocess the IMDB dataset
# ----------------------------
num_words = 10000   # Keep top 10,000 most frequent words
maxlen = 100        # Max length of each review

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Pad sequences to ensure equal length
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# ----------------------------
# Step 2: Build the LSTM model
# ----------------------------
model = Sequential([
    Embedding(num_words, 128, input_length=maxlen),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary output: Positive / Negative
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ----------------------------
# Step 3: Train the model
# ----------------------------
history = model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.2, verbose=1)

# ----------------------------
# Step 4: Evaluate the model
# ----------------------------
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("\n🧾 Model Evaluation Results:")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy*100:.2f}%")

# ----------------------------
# Step 5: Print Sample Inputs and Predictions
# ----------------------------
# Decode words back from integers
word_index = imdb.get_word_index()
reverse_word_index = {value + 3: key for key, value in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"
reverse_word_index[3] = "<UNUSED>"

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# Display 3 random samples from test set
for i in range(3):
    sample_input = x_test[i]
    print(f"\n🎞️ Sample Review {i+1}:")
    print(decode_review(sample_input))
    
    prediction = model.predict(np.expand_dims(sample_input, axis=0), verbose=0)
    sentiment = "😊 Positive" if prediction[0][0] > 0.5 else "😡 Negative"
    print(f"🔮 Predicted Sentiment: {sentiment}")
    print(f"🧠 Actual Sentiment: {'😊 Positive' if y_test[i]==1 else '😡 Negative'}")

# ----------------------------
# Step 6: Explain the Working of LSTM
# ----------------------------
print("\n🧩 Explanation:")
print("LSTM (Long Short-Term Memory) networks are a type of RNN that can remember long-term dependencies.")
print("They capture sequential context in text — understanding that the meaning of words depends on their order.")
print("For example, ‘not good’ is correctly recognized as negative because LSTM remembers the effect of ‘not’.")

