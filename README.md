# NLP Text Summarization Research

This repository hosts the implementation of an NLP model for abstractive text summarization, designed as part of a research project. 
The model architecture was constructed in TensorFlow, based on specifications from the Pointer Generator paper. 
A key improvement of this implementation is a vectorized beam search algorithm, coupled with a dataset preprocessing pipeline that is optimized for GPU acceleration.

## Highlights

- **TensorFlow Implementation**: Model built from the ground up using TensorFlow, ensuring efficient computation and model training.
- **Optimized Beam Search**: Customized beam search algorithm vectorized for better performance during the inference phase.
- **GPU-Optimized Preprocessing**: The preprocessing pipeline is tailored to leverage GPU acceleration, reducing data preparation time significantly.
- **Czech Articles Dataset**: The model was rigorously tested on a dataset comprising Czech language articles, catering to the unique challenges of this language.
- **State-of-the-Art Performance**: Achieved a new benchmark by surpassing the existing State-of-the-Art (SOTA) by 2.6 ROGUE points.
- **Award-Winning Research**: The research paper detailing this work was awarded 5th place in the SOC, a prestigious science competition for high school students in the Czech Republic.

## Repository Structure

- `/model`: Contains the TensorFlow model files and architecture.
- `/helper_funcs.py`: Helper functions to assist in the model's operation.
- `/rouge_eval.py`: Evaluation script for calculating ROGUE scores.
- `/word_dict.pickle`: Serialized word dictionary for the model's vocabulary.
- Files starting with `sp_czechsum_50K` pertain to the model trained on the Czech articles dataset.

## Getting Started

To get started with this project, clone the repository and install the required dependencies listed in `requirements.txt`.

