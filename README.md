# Comparison of Deep Learning Models for Molecular Property Prediction Using the ZINC Dataset

## 1. Objective
The primary goal of this work is to compare different deep learning models for molecular property prediction using the ZINC dataset.

## 2. Methodology
**Dataset**: The ZINC dataset is a collection of commercially available chemical compounds. It provides molecular structures and associated properties, which are used for training and evaluating the models.

**Model Descriptions**:
- **Model 1: GIN (Graph Isomorphism Network)**:
  - A GIN is designed to predict molecular properties directly from the graph representation of molecules.
- **Model 2: Pretrained and Fine-tuned Model**:
  - This model is first pretrained on a node classification task to learn general features of molecular graphs.
  - It is then fine-tuned specifically for the property prediction task.
- **Model 3: Multitask Learning Model**:
  - This model simultaneously learns two tasks: node classification and property prediction.
  - It shares some components between the tasks to leverage multitask learning benefits.

**Code Attribution**: This code is adapted from [deepfindr/gnn-project](https://github.com/deepfindr/gnn-project).


## 3. Experimental Setup
**Training and Validation**:
- The training procedure includes splitting the data into training, validation, and test sets.
- Hyperparameters, loss functions, and optimization techniques are kept consistent across models for fair comparison.

**Evaluation Metrics**:
- The primary evaluation metric is Mean Squared Error (MSE) for property prediction.

## 4. Results
**Performance Comparison**:
- The following table summarizes the MSE results for all three models:

| Model                                | Mean Squared Error (MSE) |
|--------------------------------------|--------------------------|
| GIN                                  | 0.133                    |
| Pretrained and Fine-tuned            | 0.120                    |
| Multitask Learning                   | 0.143                    |

- The pretrained and fine-tuned model achieved the lowest MSE.


