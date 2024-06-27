import matplotlib.pyplot as plt
import pickle

import mlflow
mlflow.set_tracking_uri("http://localhost:5011")
import os
# run in terminal
#mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5011



def read_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data
path =  'zinc_06_13_128/zinc_06_26'#'zinc_multitask'#'property_prediction_06_26'#
multitask = False

# Read the data from the files
try :
    train_losses = read_data(f'{path}/train_losses.txt')
    train_mses = read_data(f'{path}/train_mses.txt')
    train_r2s = read_data(f'{path}/train_r2s.txt')
    valid_losses = read_data(f'{path}/test_losses.txt')
    valid_mses = read_data(f'{path}/test_mses.txt')
    valid_r2s = read_data(f'{path}/test_r2s.txt')
    if multitask:
        test_accuracies = read_data(f'{path}/test_accuracies.txt')
    
        train_accuracies = read_data(f'{path}/train_accuracies.txt')
except:
    valid_losses= read_data(f'{path}/valid_losses.txt')
    valid_mses = read_data(f'{path}/valid_mses.txt')
    valid_r2s = read_data(f'{path}/valid_r2s.txt')


# Plotting the data
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, valid_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(epochs, train_mses, label='Train MSE')
plt.plot(epochs, valid_mses, label='Test MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(epochs, train_r2s, label='Train R2')
plt.plot(epochs, valid_r2s, label='Test R2')
plt.xlabel('Epochs')
plt.ylabel('R2')
plt.legend()

if multitask:
    plt.subplot(4, 1, 4)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

plt.tight_layout()
plt.savefig('training_Test_metrics.png')
plt.show()

# Log the data to MLflow
hyperparameters = {
    "Pretrained": True,#"Multitask":  True, #, "Property": True,
    "learning_rate": 0.001,
    "batch_size": 64,
    "num_epochs": 3,
    "hidden_channels": 128,
    "num_layers": 3,
    "num_classes": 9,
    "out_channels": 3
}

with mlflow.start_run():

    for param_name, param_value in hyperparameters.items():
        mlflow.log_param(param_name, param_value)
    for epoch in epochs:
    # Log metrics with three decimal places
        mlflow.log_metric("Train Loss", round(train_losses[epoch-1], 3), step=epoch)
        mlflow.log_metric("Test Loss", round(valid_losses[epoch-1], 3), step=epoch)
        mlflow.log_metric("Train MSE", round(train_mses[epoch-1], 3), step=epoch)
        mlflow.log_metric("Test MSE", round(valid_mses[epoch-1], 3), step=epoch)
        mlflow.log_metric("Train R2", round(train_r2s[epoch-1], 3), step=epoch)
        mlflow.log_metric("Test R2", round(valid_r2s[epoch-1], 3), step=epoch)
        if multitask:
            mlflow.log_metric("Train Accuracy", round(train_accuracies[epoch-1], 3), step=epoch)
            mlflow.log_metric("Test Accuracy", round(test_accuracies[epoch-1], 3), step=epoch)

            mlflow.log_metric("Test Accuracy", test_accuracies[epoch-1], step=epoch)

    mlflow.log_artifact('training_Test_metrics.png')
   


print('To see the results in mlflow check http://localhost:5011')
