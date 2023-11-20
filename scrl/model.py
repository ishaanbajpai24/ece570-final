import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class LinearTokenSelector(nn.Module):
    def __init__(self, encoder, embedding_size=768):
        super(LinearTokenSelector, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(embedding_size, 2, bias=False)

    def forward(self, x):
        output = self.encoder(x)
        x = output[0]
        x = self.classifier(x)
        x = F.log_softmax(x, dim=2)
        return x

    def save(self, classifier_path, encoder_path):
        state = self.state_dict()
        state = dict((k, v) for k, v in state.items() if k.startswith("classifier"))
        torch.save(state, classifier_path)
        self.encoder.save_pretrained(encoder_path)

def prepare_inference_data(texts, tokenizer, max_length=512):
    encoded_inputs = tokenizer.batch_encode_plus(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt'
    )
    return encoded_inputs['input_ids'], encoded_inputs['attention_mask']

def labels_to_summary(input_ids, label_batch, tokenizer):
    summaries = []
    for input_ids, labels in zip(input_ids, label_batch):
        selected = [int(input_ids[i]) for i in range(len(input_ids)) if labels[i] == 1]
        summary = tokenizer.decode(selected, skip_special_tokens=True)
        summaries.append(summary)
    return summaries

def batch_predict(model, texts, tokenizer, device, batch_size=16):
    input_ids, attention_mask = prepare_inference_data(texts, tokenizer)
    input_ids = torch.tensor(input_ids).to(device)
    attention_mask = torch.tensor(attention_mask).to(device)
    
    num_samples = len(texts)
    predictions = []

    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_input_ids = input_ids[i:i + batch_size]
            logits = model.forward(batch_input_ids)
            argmax_labels = torch.argmax(logits, dim=2)
            batch_predictions = labels_to_summary(batch_input_ids, argmax_labels, tokenizer)
            predictions.extend(batch_predictions)

    return predictions

def load_model(model_dir, device="cuda", prefix="best"):
    if isinstance(model_dir, str):
        model_dir = Path(model_dir)
    for p in (model_dir / "checkpoints").iterdir():
        if p.name.startswith(f"{prefix}"):
            checkpoint_dir = p
    return load_checkpoint(checkpoint_dir, device=device)

def load_checkpoint(checkpoint_dir, device="cuda"):
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = Path(checkpoint_dir)

    encoder_path = checkpoint_dir / "encoder.bin"
    classifier_path = checkpoint_dir / "classifier.bin"

    encoder = AutoModel.from_pretrained(encoder_path).to(device)
    embedding_size = encoder.config.hidden_size

    classifier = LinearTokenSelector(None, embedding_size).to(device)
    classifier_state = torch.load(classifier_path, map_location=device)
    classifier_state = dict(
        (k, v) for k, v in classifier_state.items()
        if k.startswith("classifier")
    )
    classifier.load_state_dict(classifier_state)
    classifier.encoder = encoder
    return classifier.to(device)

def parallel_predict(texts, tokenizer, model, device, num_workers=4):
    results = []
    batch_size = len(texts) // num_workers

    def predict_batch(batch_texts):
        return batch_predict(model, batch_texts, tokenizer, device)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(predict_batch, texts[i:i+batch_size]) for i in range(0, len(texts), batch_size)]
        for future in tqdm(futures, total=len(futures), desc="Predicting"):
            results.extend(future.result())

    return results

