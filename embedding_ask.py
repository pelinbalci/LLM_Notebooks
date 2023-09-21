import ast
import pandas as pd
from embedding_search import similarity

# read embeddings
df = pd.read_pickle('article_embeddings.pkl')

# query
query = "what is gender equality"

strings, relatednesses = similarity(query, df, top_n=2)
for string, relatedness in zip(strings, relatednesses):
    print(f"{relatedness=:.3f}")
    print(string)


# Prepare prompt
strings, relatednesses = similarity(query, df)

# message to model
introduction = 'Use the below articles on the gender equality to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'

# user's question
question = f"\n\nQuestion: {query}"


message = introduction

for string in strings:
    next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
    if len((message + next_article + question).split()) > 300:
        break
    else:
        message += next_article

final_query = message + question

# ----------------------------------------------------------------------------------------------------------------------
# ASK

from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

# final_query = "I liked 'Breaking Bad' and 'Band of Brothers'. Do you have any recommendations of other shows I might like?"
sequences = pipeline(
    final_query,
    do_sample=True,
    top_k=10,
    top_p=0.9,
    temperature=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=1024,  # can increase the length of sequence
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

print('--------------------------------------------------')
print(sequences[0]['generated_text'][len(final_query):])

# Use the below articles on the gender equality to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."
# Wikipedia article section:
# """
# Gender equality is the goal, while gender neutrality and gender equity are practices and ways of thinking that help in achieving the goal.
# """
# Wikipedia article section:
# """
# The Council of Europe's Convention on preventing and combating violence against women and domestic violence, the first legally binding instrument in Europe in the field of violence against women,[14] came into force in 2014.
# """
# Question: what is gender equality according to the Wikipedia articles?
# Choose one of the following options:
# A. Gender equality is the goal, while gender neutrality and gender equity are practices and ways of thinking that help in achieving the goal.
# B. Gender equality is the practice of ensuring that both men and women have equal rights and opportunities.
# C. Gender equality is the process of creating a fair and just society where all individuals are treated with dignity and respect regardless of their gender.
# D. I could not find an answer.
#