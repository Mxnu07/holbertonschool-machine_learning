import GPyOpt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
import matplotlib.pyplot as plt

def load_data():
    print("Loading data...")
    (X_train, y_train), (X_val, y_val) = tf.keras.datasets.mnist.load_data()
    X_train, X_val = X_train / 255.0, X_val / 255.0
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    (X_test, y_test) = tf.keras.datasets.mnist.load_data()[1]
    X_test = X_test / 255.0
    X_test = X_test[..., np.newaxis]

    print("Data loaded.")
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_cnn_model(learning_rate, filters, kernel_size, dropout_rate, l2_reg, batch_size):
    print(f"Building model with lr={learning_rate}, filters={filters}, kernel_size={kernel_size}, dropout_rate={dropout_rate}, l2_reg={l2_reg}, batch_size={batch_size}")
    model = Sequential()
    model.add(Conv2D(filters, kernel_size, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg), input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def objective_function(params):
    learning_rate, filters, kernel_size, dropout_rate, l2_reg, batch_size = params[0]
    model = build_cnn_model(learning_rate, int(filters), int(kernel_size), dropout_rate, l2_reg, int(batch_size))

    checkpoint_path = f"model_lr{learning_rate}_filters{filters}_kernel{kernel_size}_dropout{dropout_rate}_l2{l2_reg}_batch{batch_size}.keras"
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=int(batch_size), callbacks=[checkpoint, early_stopping], verbose=0)

    best_accuracy = max(history.history['val_accuracy'])
    return -best_accuracy  # Minimize negative accuracy

def main():
    print("Starting main function...")
    global X_train, y_train, X_val, y_val, X_test, y_test
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Define the hyperparameter space
    bounds = [
        {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-5, 1e-1)},
        {'name': 'filters', 'type': 'discrete', 'domain': (32, 64, 128)},
        {'name': 'kernel_size', 'type': 'discrete', 'domain': (3, 5, 7)},
        {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0.0, 0.5)},
        {'name': 'l2_reg', 'type': 'continuous', 'domain': (1e-5, 1e-2)},
        {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64, 128)}
    ]

    # Initialize Bayesian Optimization
    print("Initializing Bayesian Optimization...")
    optimizer = GPyOpt.methods.BayesianOptimization(f=objective_function, domain=bounds, maximize=False)


    # Run optimization
    print("Running optimization...")
    optimizer.run_optimization(max_iter=30)

    # Plot convergence
    print("Plotting convergence...")
    optimizer.plot_convergence()
    plt.savefig('convergence_plot.png')

    # Save optimization report
    print("Saving optimization report...")
    with open('bayes_opt.txt', 'w') as f:
        f.write(f"Best hyperparameters: {optimizer.x_opt}\n")
        f.write(f"Best accuracy: {-optimizer.fx_opt}\n")

    # Load the best model
    print("Loading the best model...")
    best_model_path = f"model_lr{optimizer.x_opt[0]}_filters{optimizer.x_opt[1]}_kernel{optimizer.x_opt[2]}_dropout{optimizer.x_opt[3]}_l2{optimizer.x_opt[4]}_batch{optimizer.x_opt[5]}.keras"
    best_model = tf.keras.models.load_model(best_model_path)

    # Evaluate the best model
    print("Evaluating the best model...")
    best_model.evaluate(X_test, y_test)

    print("Done.")
if __name__ == "__main__":
    main()
