import sys
sys.path.append('..')
import os 
from common.np import *

def clip_grads(grads, max_norm):
    """
    grad 가 max_norm 보다 클 경우
    rate를 곱해서 max 크기로 제한
    """
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)

    if rate < 1:
        for grad in grads:
            grad *= rate

def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, vocab_size, window_size=1):
    """
    co-occurence matrix(동시 발생 행렬) 생성

    Parameters
    -----------
    corpus : List[str]
        말뭉치 (단어 id 목록)
    vocab_size : int
        어휘 수
    window_size : int
        윈도우 크기 (n의 윈도우 크기 => 집중하는 단어의 양 옆으로 n개의 단어가 맥락에 포함)
    
    
    Return
    -----------
    co_matrix : np.array([int])
        co-occurence matrix
    """

    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size +  1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    
    return co_matrix

def cos_similarity(x, y, eps=1e-8):
    '''
    cosine similarity

    Parameters
    ---------------
    x : np.array
        벡터
    y : np.array
        벡터
    eps : 0으로 나누게 되는 경우 (x 또는 y 가 0일 경우) 무한대가 나오는 에러 방지

    Return
    -------
    코사인 유사도 : np.array
    '''

    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y **2)) + eps)

    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''
    유사 단어 검색

    Parameters
    ------------
    query : str
        기준이 되는 텍스트
    word_to_id : Dict
        단어에서 단어 ID로 반환하는 딕셔너리
    id_to_word : Dict
        단어 ID에서 단어로 변환하는 딕셔너리
    word_matrix : np.array 
        단어 벡터를 정리한 행렬, 각 행에 해당 단어 벡터 저장
    top : int
        상위 단어 몇 개까지 출력할 지 지정
    '''

    if query not in word_to_id:
        print(f"{query}를 찾을 수 없습니다.")
        return
    
    print(f'\n[query] {query}')
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)

    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    count = 0
    for i in (-1 * similarity).argsort(): # similarity에 -1을 곱하여 가장 큰 similarity가 가장 먼저 뽑히게 설정
        if id_to_word[i] == query: # 자기 자신
            continue
            
        print(f'{id_to_word[i]}: {similarity[i]}')

        count += 1

        if count >= top:
            return


def ppmi(C, verbose=False, eps=1e-8):
    '''
    Postive pointwise mutual information을 구한다.
    co-ocurrence matrix를 통해 근사값을 구하도록 구성(중복된 횟수가 존재하므로 정확한 값 x)
    '''
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[i] * S[j] + eps))
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print(f"{100 * cnt / total: .2f} 완료")
    
    return M





