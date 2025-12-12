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

class PairItem:
    def __init__(self,count,first_b,second_b):
        self.count=count
        self.first_b=first_b
        self.second_b=second_b

    def __lt__(self,other):
        if self.count != other.count:
            return self.count>other.count
        if self.first_b != other.first_b:
            return self.first_b>other.first_b
        return self.second_b>other.second_b

class tokenizer_training:
    def __init__(self,vocab_size: int, special_tokens: list[str]):
        #useful data structures
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        self.special_tokens_bytes = [ token.encode('utf-8') for token in self.special_tokens ]
        self.vocab_dict: Dict[int, bytes] = {}
        self.vocab_dict_reverse: Dict[bytes, int] = {}
        self.merges: List[Tuple[bytes,bytes]] = []

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
        pair_to_words: Dict[(int,int),Set[bytes]] = {}
        for word in word_set:
            word_ids=word_to_ids[word]
            if len(word_ids)>1:
                for i in range(len(word_ids)-1):
                    pair=(word_ids[i],word_ids[i+1])
                    pair_count[pair]=pair_count.get(pair,0)+word_count[word]
                    if pair in pair_to_words.keys():
                        pair_to_words[pair].add(word)
                    else:
                        pair_to_words[pair]={word}
        for pair,count in pair_count.items():
            heap_item=PairItem(count,self.vocab_dict[pair[0]],self.vocab_dict[pair[1]])
            heapq.heappush(heap,heap_item)

        #merging
        while len(self.vocab_dict)<self.vocab_size:
            #find the best pair and update global stuff
            found=False
            while heap:
                curr_item=heapq.heappop(heap)
                count, bytes_1, bytes_2 = curr_item.count,curr_item.first_b,curr_item.second_b
                new_bytes = bytes_1 + bytes_2
                new_pair = (self.vocab_dict_reverse[bytes_1], self.vocab_dict_reverse[bytes_2])
                if count==pair_count.get(new_pair, 0) and count>0:
                    found=True
                    break
            if not found:
                break
            self.vocab_dict[len(self.vocab_dict)] = new_bytes
            self.vocab_dict_reverse[new_bytes] = len(self.vocab_dict_reverse)
            self.merges.append((bytes_1,bytes_2))
            new_id=self.vocab_dict_reverse[new_bytes]

            #merge the words and update word_to_ids and heap and pair_to_words and pair_count
            candidates=pair_to_words[new_pair].copy()
            for word in candidates:
                word_ids=word_to_ids[word]
                if len(word_to_ids[word])<2:
                    continue
                new_word_ids=[]
                i=0
                while i <len(word_ids)-1:
                    curr_pair=(word_ids[i],word_ids[i+1])
                    if curr_pair==new_pair:
                        new_word_ids.append(new_id)
                        i+=2
                    else:
                        new_word_ids.append(word_ids[i])
                        i+=1
                if i==len(word_ids)-1:
                    new_word_ids.append(word_ids[i])

                #update word_to_ids
                word_to_ids[word]=new_word_ids

                #count old and new pairs
                old_pair_count = {}
                new_pair_count = {}
                affected_pairs=set()
                for i in range(len(word_ids)-1):
                    curr_pair = (word_ids[i], word_ids[i+1])
                    old_pair_count[curr_pair]= old_pair_count.get(curr_pair,0) +1
                for i in range(len(new_word_ids)-1):
                    curr_pair = (new_word_ids[i], new_word_ids[i+1])
                    new_pair_count[curr_pair]= new_pair_count.get(curr_pair,0) +1

                #update pair_to_words and pair_count
                for pair, count in old_pair_count.items():
                    if pair not in new_pair_count.keys():
                        affected_pairs.add(pair)
                        pair_to_words[pair].remove(word)
                        pair_count[pair]-= count*word_count[word]
                    else:
                        if new_pair_count[pair]!=count:
                            affected_pairs.add(pair)
                            pair_count[pair] += word_count[word]*(new_pair_count[pair]-count)

                for pair, count in new_pair_count.items():
                    if pair not in old_pair_count.keys():
                        affected_pairs.add(pair)
                        if pair in pair_to_words.keys():
                            pair_to_words[pair].add(word)
                        else:
                            pair_to_words[pair]={word}
                        pair_count[pair] = pair_count.get(pair,0)+ word_count[word] * count

                #update heap
                for pair in affected_pairs:
                    new_item=PairItem(pair_count[pair],self.vocab_dict[pair[0]],self.vocab_dict[pair[1]])
                    heapq.heappush(heap,new_item)

        return self.vocab_dict, self.merges

class tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab_inverse={v:k for k,v in self.vocab.items()}

        if self.special_tokens is not None:
            special_tokens_bytes = set([token.encode("utf-8") for token in self.special_tokens])

            for byte_token in special_tokens_bytes:
                if byte_token not in self.vocab.values():
                    self.vocab[len(self.vocab)] = byte_token

    def get_pairs(self, symbols):
        """Return set of adjacent symbol pairs in the current word."""
        pairs = set()
        for i in range(len(symbols) - 1):
            pairs.add((symbols[i], symbols[i + 1]))
        return pairs

    def encode(self, text: str) -> list[int]:

        output_ids = []

        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = f"({'|'.join(re.escape(s) for s in sorted_special_tokens)})"
            word_chunks = re.split(special_pattern, text)
        else:
            word_chunks = [text]

        for chunk in word_chunks:

            if self.special_tokens is not None and chunk in self.special_tokens:
                output_ids.append(self.vocab_inverse[chunk.encode("utf-8")])
                continue

            words_in_bytes=pre_tokenize(chunk)
            for word in words_in_bytes:
                symbols = [bytes([b]) for b in word]

                if not symbols:
                    continue

                pairs=self.get_pairs(symbols)

                while True:
                    best_pair=None
                    best_rank=None
                    for pair in pairs:
                        rank=self.merge_ranks.get(pair)
                        if rank is not None and (best_rank is None or rank<best_rank):
                            best_rank=rank
                            best_pair=pair

                    if best_pair is None:
                        break

                    new_symbols=[]
                    i=0
                    while i<len(symbols):
                        if i<len(symbols)-1 and (symbols[i], symbols[i+1])==best_pair:
                            new_symbols.append(symbols[i]+ symbols[i+1])
                            i+=2
                        else:
                            new_symbols.append(symbols[i])
                            i+=1

                    symbols=new_symbols
                    pairs=self.get_pairs(symbols)

                for b in symbols:
                    output_ids.append(self.vocab_inverse[b])

        return output_ids

    def encode_iterable(self, iterable):
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        all_bytes=b"".join(self.vocab[num] for num in ids)
        return all_bytes.decode("utf-8", errors="replace")

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab_dict = dict()
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                curr_combo = line.strip().split('\t')
                if len(curr_combo) == 2:
                    curr_id = int(curr_combo[0])
                    curr_bytes = bytes.fromhex(curr_combo[1])
                    vocab_dict[curr_id] = curr_bytes

        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                curr_combo = line.strip().split('\t')
                if len(curr_combo) == 2:
                    curr_bytes_1 = bytes.fromhex(curr_combo[0])
                    curr_bytes_2 = bytes.fromhex(curr_combo[1])
                    merges.append((curr_bytes_1, curr_bytes_2))

        return cls(vocab_dict, merges, special_tokens)
