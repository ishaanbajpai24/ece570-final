# Optimizing Text Classification: Insights and Outcomes from Sentence Summarization Techniques 

Sentence Compression with Reinforcement Learning

## Code Files Changed and Re-Implemented
* **model.py:** 

    * **Added attention masks:** This addition was aimed at improving the model’s ability to focus on relevant parts of the input, which is especially crucial for tasks involving sentence compression. (functions: `prepare_inference_data` and `batch_predict`)
    * **Introduction of Concurrent Processing:** This function standardizes and optimizes the preparation of data for model inference. (function: `parallel_predict`)
    * **Optimized Batch Prediction:** Batch processing reduces the computational burden on the system, especially when dealing with large volumes of data. It efficiently manages memory usage and speeds up the inference process by taking advantage of vectorized operations in PyTorch. (function: `batch_predict`)
    * **Refinement of Model Loading and State Restoration:** These refinements ensure seamless integration of the model loading process with the new parallel processing and batch prediction functionalities, maintaining consistency and reliability. (functions: `load_model` and `load_checkpoint`)

    * **Optimization of Text Padding and Truncation:** Implemented optimized strategies for text padding and truncation. These modifications were intended to enhance the model’s processing capabilities. (functions: `prepare_inference_data`)

* **main.py:**

    * Modified the flow of the code
    * Added memory usage tracking
    * Added inference time evaluation 
## Installation

Install `scrl` library and other dependencies

```bash
pip install scrl
pip install -r requirements.txt <br>
pip install -e .
```

## Data
The data being used for this project can be found in the following Google Drive. 

`https://drive.google.com/drive/folders/1grkgZhtdd-Bw45GAnHza9RRb5OVQG4pK?usp=sharing`

Models are required to use and evaluate our trained models

For this particular implementation, I have used `newsroom-P75` which is trained to reduce sentences to 75% of their original length
## Usage

To run the experiment I have set up, execute the python file `main.py`. This can be done using the command:
```bash
python main.py
```
The input text that is currently set up to go for inference is: 
```
As the sun set over the horizon, painting the sky in hues of orange and pink, the bustling city slowly transformed into
a serene landscape, where the distant sounds of traffic melded with the gentle rustling of leaves, creating a symphony 
of urban life that resonated with the rhythmic heartbeat of the metropolis, reminding everyone of the intricate balance 
between nature and civilization, and the beauty that emerges when these two forces coexist in harmony, each lending its 
unique character to the tapestry of existence.
```
The expected output for this input is:
```
As the sun set over the horizon, painting the sky in hues of orange and pink, the bustling city slowly transformed into 
a serene landscape, where sounds of traffic melded with rustling leaves, creating a symphony of urban life that 
resonated with rhythmic the metropolis, reminding balance between nature and civilization and beauty when these two 
forces coexist in harmony, lending to tapestry existence.
```

This shows a compression from 85 words to 64 words, without losing important information in the process.

This input text can be changed by going into the `main.py` file and changing the text being sent in.

### Citation

```
@inproceedings{ghalandari-etal-2022-efficient,
    title = "Efficient Unsupervised Sentence Compression by Fine-tuning Transformers with Reinforcement Learning",
    author = "Gholipour Ghalandari, Demian and Hokamp, Chris and Ifrim, Georgiana",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2205.08221",
    pages = "1267--1280",
}
```