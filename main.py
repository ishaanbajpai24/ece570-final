from scrl.model import load_model, batch_predict
from transformers import AutoTokenizer
import time
import tracemalloc


def format_summaries(summaries):
        return "\n".join(summaries)

def main(model_dir, cpu, encoder_model, input_texts):

    model = load_model(model_dir, device)
    tokenizer = AutoTokenizer.from_pretrained(encoder_model)
    input_texts = [x.strip() for x in input_texts]
    tracemalloc.start()
    
    start = time.time()
    summaries = batch_predict(model, input_texts, tokenizer, device)
    end = time.time()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
    
    out_summaries = format_summaries(summaries)

    return out_summaries, end-start

if __name__ == '__main__':
    model_dir = "data/models/newsroom-P75/"
    device = "cpu"
    encoder_model = "distilroberta-base"
    input_texts = [
    """
    As the sun set over the horizon, painting the sky in hues of orange and pink, the bustling city slowly transformed into a serene landscape, where the distant sounds of traffic melded with the gentle rustling of leaves, creating a symphony of urban life that resonated with the rhythmic heartbeat of the metropolis, reminding everyone of the intricate balance between nature and civilization, and the beauty that emerges when these two forces coexist in harmony, each lending its unique character to the tapestry of existence.    
    """
    ]

    out_summaries, time_taken = main(model_dir, device, encoder_model, input_texts)

    print()
    print(f"Time taken {time_taken: .4f} seconds")
    print()
    print(out_summaries)
