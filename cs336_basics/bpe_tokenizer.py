import regex as re
from pathlib import Path
from typing import Iterable, Iterator, List, Dict, Tuple, Set
import heapq

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenize(text:str) -> List[bytes]:
    str_tokens = re.findall(PAT, text)
    bytes_tokens = [token.encode("utf-8") for token in str_tokens]
    return bytes_tokens

class Node:
    def __init__(self,value):
        self.value=value
        self.prev=None
        self.next=None

class BPETokenizer:
    def __init__(self,vocab_size: int, special_tokens: list[str]):
        #useful data structures
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        self.special_tokens_bytes = [ token.encode('utf-8') for token in self.special_tokens ]
        self.vocab_dict: Dict[int, bytes] = {}
        self.vocab_dict_reverse: Dict[bytes, int] = {}
        self.merges: List[Tuple[bytes,bytes]] = []
        self.merges_rank: Dict[Tuple[bytes,bytes], int] = {}

        #initialize vocab dicts
        for i in range(256):
            self.vocab_dict[len(self.vocab_dict)]=bytes([i])
            self.vocab_dict_reverse[bytes([i])]=len(self.vocab_dict_reverse)

        for token in self.special_tokens_bytes:
            self.vocab_dict[len(self.vocab_dict)]=token
            self.vocab_dict_reverse[token]=len(self.vocab_dict_reverse)

    def train_tokenizer(self, path:str):

        #open file as str
        with open(path, "r", encoding="utf-8") as f:
            text=f.read()

        #protect special tokens, spilt text as chunks
        if self.special_tokens:
            special_pattern = f"({'|'.join(re.escape(s) for s in self.special_tokens)})"
            text_chunks = re.split(special_pattern, text)
        else:
            text_chunks = [text]

        #initialize the word as ids
        word_set: Set[bytes] = set()
        word_count: Dict[bytes, int] = {}
        word_to_ids: Dict[bytes, List[int]] = {}

        for chunk in text_chunks:
            if chunk in self.special_tokens or not chunk:
                continue
            words_in_bytes=pre_tokenize(chunk)
            for word in words_in_bytes:
                word_set.add(word)
                if word in word_count:
                    word_count[word]+=1
                else:
                    word_count[word]=1

        for word in word_set:
            word_id_list: List[int] = [self.vocab_dict_reverse[bytes([char])] for char in word]
            word_to_ids[word]=word_id_list

        #initialize pairs
        heap=[]
        pair_count: Dict[(int,int),int] = {}
        words_pair_count: Dict[bytes,Dict[(int,int),int]] = {}
        for word in word_set:
            word_ids=word_to_ids[word]
            if len(word_ids)>1:
                words_pair_count[word] = {}
                for i in range(len(word_ids)-1):
                    pair=(word_ids[i],word_ids[i+1])
                    pair_count[pair]=pair_count.get(pair,0)+word_count[word]
                    words_pair_count[word][pair]=words_pair_count[word].get(word,0)+1
        for pair,count in pair_count.items():
            heap_item=(-count,self.vocab_dict[pair[0]],self.vocab_dict[pair[1]])
            heapq.heappush(heap,heap_item)

        #merging
        merge_rank=0
        while len(self.vocab_dict)<self.vocab_size:
            #find the best pair and update global stuff
            found=False
            while heap:
                neg_count, bytes_1, bytes_2 = heapq.heappop()
                new_bytes = bytes_1 + bytes_2
                new_pair = (self.vocab_dict_reverse[bytes_1], self.vocab_dict_reverse[bytes_2])
                if -neg_count==pair_count.get(new_pair, 0):
                    found=True
                    break
            if not found:
                break
            self.vocab_dict[len(self.vocab_dict)] = new_bytes
            self.vocab_dict_reverse[new_bytes] = len(self.vocab_dict_reverse)
            self.merges.append((bytes_1,bytes_2))
            self.merges_rank[(bytes_1, bytes_2)]=merge_rank
            new_id=self.vocab_dict_reverse[new_bytes]
            merge_rank+=1

            #merge the words and update word_to_ids and heap and words_pair_count and pair_count
            for word in word_set:
                if new_pair in words_pair_count[word].keys():
                    affected_pairs = set()
                    break_points = []
                    word_ids=word_to_ids[word]
                    i=0
                    while i<len(word_ids)-1:
                        curr_pair=(word_ids[i],word_ids[i+1])
                        if curr_pair==new_pair:
                            break_points.append(i)

                            pair_count[new_pair]-=word_count[word]
                            affected_pairs.add(new_pair)
                            words_pair_count[word][new_pair]-=1
                            if words_pair_count[word][new_pair]==0:
                                words_pair_count[word].pop(new_pair)

                            if i>0:
                                prev_pair=(word_ids[i-1],word_ids[i])
                                pair_count[prev_pair] -= word_count[word]
                                affected_pairs.add(prev_pair)
                                words_pair_count[word][prev_pair] -= 1
                                if words_pair_count[word][prev_pair] == 0:
                                    words_pair_count[word].pop(prev_pair)

                                new_prev_pair=(word_ids[i-1],new_id)
                                pair_count[new_prev_pair] = pair_count.get(new_prev_pair,0)+ word_count[word]
                                affected_pairs.add(new_prev_pair)
                                words_pair_count[word][new_prev_pair]=words_pair_count[word].get(new_prev_pair,0)+1

                            if i+2<len(word_ids):
                                next_pair=(word_ids[i+1],word_ids[i+2])
                                pair_count[next_pair] -= word_count[word]
                                affected_pairs.add(next_pair)
                                words_pair_count[word][next_pair] -= 1
                                if words_pair_count[word][next_pair] == 0:
                                    words_pair_count[word].pop(next_pair)

                                new_next_pair = (new_id, word_ids[i+2])
                                pair_count[new_next_pair] = pair_count.get(new_next_pair, 0) + word_count[word]
                                affected_pairs.add(new_next_pair)
                                words_pair_count[word][new_next_pair] = words_pair_count[word].get(new_next_pair, 0) + 1

                            i+=2
                        else:
                            i+=1

                    for pair in affected_pairs:
                        new_item=(-pair_count[pair],self.vocab_dict[pair[0]],self.vocab_dict[pair[1]])
                        if -pair_count[pair]>0:
                            heapq.heappush(heap,new_item)

                    new_word_ids=word_ids[:break_points[0]]
                    for i in range(len(break_points)-1):
                        s=break_points[i]
                        t=break_points[i+1]
                        new_word_ids.append(new_id)
                        new_word_ids+=word_ids[s+2:t]
                    new_word_ids.append(new_id)
                    new_word_ids+=word_ids[break_points[-1]+2:]

                    word_to_ids[word]=new_word_ids

        return self.vocab_dict, self.merges, self.merges_rank




