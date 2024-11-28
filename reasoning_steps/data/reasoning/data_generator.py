from datasets import load_dataset
import re
import nltk
import pycld2 as cld2
import pandas as pd
from tqdm import tqdm
import random
from transformers import AutoTokenizer
import seaborn as sns
from matplotlib import pyplot as plt

#num_examples = 10000
min_tokens_per_sentence = 5
max_tokens_per_sentence = 64
prompt_sentence_len = (1, 5)
prompt_sentence_len_probs = [0.4, 0.3, 0.2, 0.2, 0.3]

random.seed(27)
nltk.download("popular")
#dataset = load_dataset("Skylion007/openwebtext")["train"]
dataset = load_dataset("Lyte/Reasoning-Paused-Combined")["train"]
num_examples = len(dataset)
data_indices = list(range(len(dataset)))
random.shuffle(data_indices)
#tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")

prompts = []
stats = []
pbar = tqdm(total=num_examples)
for data_idx in data_indices:
    passage = dataset[data_idx]["text"].strip()
    passage = nltk.sent_tokenize(passage)

    # Number of sentences must be in allowed range + a plausible next sampled sentence
    if len(passage) < prompt_sentence_len[1] + 1:
        continue

    # Pick a number of sentences to use
    num_sentences = random.choices(
        range(prompt_sentence_len[0], prompt_sentence_len[1] + 1),
        weights=prompt_sentence_len_probs,
        k=1,
    )[0]

    # Find num_sentences that meets the criteria, searching in random order
    start_idx = list(range(0, len(passage) - num_sentences))
    random.shuffle(start_idx)
    for idx in start_idx:
        prompt = passage[idx : idx + num_sentences]

        # Check sentence token length bounds
        prompt_tokens = tokenizer(prompt)["input_ids"]
        if any(len(tokens) > max_tokens_per_sentence for tokens in prompt_tokens):
            continue
        if any(len(tokens) < min_tokens_per_sentence for tokens in prompt_tokens):
            continue

        # Convert from list of sentences back to one string
        prompt = " ".join(prompt) + " "

        # Check sentence only uses normal characters
        if not re.match("^[a-zA-Z0-9,.!? ]*$", prompt):
            continue

        # Check sentence is in English
        isReliable, _, details = cld2.detect(prompt)
        if not isReliable or details[0][1] != "en":
            continue

        break

    # If no prompt met all the criteria, continue to the next passage
    if idx == start_idx[-1]:
        continue

    prompts.append(prompt)
    stats.append(
        {
            "num_sentences": num_sentences,
            "num_tokens": sum(len(tokens) for tokens in prompt_tokens),
        }
    )
    pbar.update(1)
    if len(prompts) == num_examples:
        break
pbar.close()
stats = pd.DataFrame(stats)

with open("data/reasoning/prompts.txt", "w") as f:
    f.write("\n".join(prompts))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.countplot(stats, x="num_sentences", color=sns.color_palette()[0], ax=axs[0])
axs[0].set_title("Number of Sentences")
sns.histplot(stats, x="num_tokens", ax=axs[1])
axs[1].set_title("Number of Tokens")
fig.tight_layout()
fig.savefig("data/reasoning/prompts_stats.png")
