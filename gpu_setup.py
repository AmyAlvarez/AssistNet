import tensorflow as tf

def setup_gpu():
    # check if GPU is available and set it up
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # enable memory growth --> avoid using all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus}")
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")
    else:
        print("No GPU detected. Continue training on CPU.")
