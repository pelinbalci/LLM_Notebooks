import re
import numpy as np
import pandas as pd
import editdistance
from sentence_transformers import SentenceTransformer
import torch


# device
device_count = torch.cuda.device_count()
if device_count > 0:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)


# Calculate the cosine similarity for two vectors
def find_cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similarity between u and v
    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)
    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    dot = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    # Avoid division by zero error
    if np.isclose(norm_u * norm_v, 0, atol=1e-32):
        return 0
    # Compute the cosine similarity
    cosine_similarity = dot / (norm_u * norm_v)

    return cosine_similarity


# Calculate the lexical similarity of the given two texts/sentences
def find_lexical_similarity(text1, text2):
    """editdistance:  the minimum number of operations required to transfer one text into the other"""
    lexical_sim = 1 - (editdistance.eval(text1, text2) / max(len(text1), len(text2)))
    return lexical_sim


def prepare_text(text):
    # lower
    text = text.lower()
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text


def chunk_article(article):
  sentences = ["Gender equality is the goal, while gender neutrality and gender equity are practices and ways of thinking that help in achieving the goal.",
               "In 1994, the twenty-year Cairo Programme of Action was adopted at the International Conference on Population and Development (ICPD) in Cairo.",
               "The United Nations Security Council Resolution 1325 (UNSRC 1325), which was adopted on 31 October 2000, deals with the rights and protection of women and girls during and after armed conflicts.",
               "The Council of Europe's Convention on preventing and combating violence against women and domestic violence, the first legally binding instrument in Europe in the field of violence against women,[14] came into force in 2014."]
  return sentences


def get_embeddings(sentences):
  model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
  list_sentence_embeddings =[]
  for sen in sentences:
    sem = (prepare_text(sen))
    sentence_embeddings = model.encode(sen)
    list_sentence_embeddings.append(sentence_embeddings)
  df = pd.DataFrame({"sentences": sentences, "embedding": list_sentence_embeddings})
  return df


def similarity(query:str, df:pd.DataFrame, top_n: int=2) -> tuple[list[str], list[float]]:
    # prepare query embeddings
    model = SentenceTransformer('emrecan/bert-base-turkish-cased-mean-nli-stsb-tr')
    query = (prepare_text(query))
    query_embeddings = model.encode(query)

    # apply cosine similarity to two vectors
    strings_and_relatednesses = [(row["sentences"], find_cosine_similarity(query_embeddings, row["embedding"])) for i, row in df.iterrows()]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)

    # transformers_similarity_score = find_cosine_similarity(sentence_embeddings[0], sentence_embeddings[1])

    # # find lexical similarity with edit distance
    # lexical_similarity = find_lexical_similarity(text1, text2)

    # # weighted average
    # similarity_score = transformers_similarity_score * 0.6 + lexical_similarity * 0.4
    return strings[:top_n], relatednesses[:top_n]


if __name__ == "__main__":
    # find sentence embeddings
    article = "mmmm"
    sentences = chunk_article(article)
    df = get_embeddings(sentences)
    df.to_pickle('article_embeddings.pkl')




