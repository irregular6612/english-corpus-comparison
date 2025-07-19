from Levenshtein import distance # distance measure function import
import numpy as np

def orthographic_N(word : str, lexicon : list[str]) -> int: # orthographic N의 정의를 이용해서 이웃의 개수 산출.
    # 길이는 고정해두었기 때문에, levenshtein에서 허용하는 건 치환만. 또한, distance=1으로, 스스로는 제외
    return sum(1 for w in lexicon if len(w)==len(word) and distance(w, word)==1) 

def OLD20(word : str, lexicon : list[str]) -> float:
    dists = sorted(distance(word, w) for w in lexicon)[:20]
    return np.mean(dists).item()