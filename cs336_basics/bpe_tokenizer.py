import os
import regex as re
import heapq
import json
import random
from tqdm import tqdm
from typing import BinaryIO, Iterable, Iterator
from multiprocessing import Pool, cpu_count
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import chain
from .utils import IndexedHeap

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

@dataclass
class BytePair:
    first: bytes
    second: bytes
    frequency: int
    pretoken: list[int]

    def __lt__(self, other):
        if self.frequency != other.frequency:
            return self.frequency < other.frequency
        return (self.first, self.second) < (other.first, other.second)
    
    def copy(self):
        return BytePair(
            first=self.first,     
            second=self.second,    
            frequency=self.frequency,
            pretoken=self.pretoken
        )

@dataclass
class Pretoken:
    count: int
    byte_pairs: deque[bytes]

class BPETokenizerTrainer:
    def __init__(self, file_path, vocab_size, special_tokens, processor_num=4, chunk_size=100000000):
        """
        初始化重要参数
        :param file_path: 训练分词器的文本文件路径
        :param vocab_size: 词表尺寸
        :param special_tokens: 特殊token列表
        :param processor_num: 进程数量
        """
        self.file_path = file_path
        self.vocab_size = vocab_size if vocab_size else 512
        self.special_tokens = special_tokens if special_tokens else []
        self.processor_num = processor_num
        self.chunk_size = chunk_size

        self.vocab:dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i in range(len(special_tokens)):
            self.vocab[256+i] = special_tokens[i].encode('utf-8')
        self.merges = []

        # dict[str, Pretoken]
        self.pre_tokens = {}
        self.pre_tokens_list = []
        # dict[BytePair key, BytePair]
        self.byte_pairs = {}
        self.byte_pairs_heap = IndexedHeap(is_min_heap=False)

    @staticmethod
    def _process_task(task_data):
        file_path, start, end, concat_special_tokens = task_data
        # Pattern for pre-tokenization
        pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        with open(file_path, 'rb') as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            splited_chunk = re.split(concat_special_tokens, chunk)
            local_counter = defaultdict(int)
            
            for seg in splited_chunk:
                for match in re.finditer(pat, seg):
                    #pre_token = tuple(match.group().encode("utf-8"))
                    pre_token = match.group()
                    local_counter[pre_token] += 1
        
        return local_counter

    def pre_tokenation(self, tasks):
        concat_special_tokens = "|".join(self.special_tokens)
        tasks = [task + (concat_special_tokens,) for task in tasks]
        with Pool(processes=self.processor_num) as pool:
            results = list(tqdm(
                pool.imap(self._process_task, tasks), 
                total=len(tasks), 
                desc="pre tokenation"
            ))
            #results = pool.map(self._process_task, tasks)
        
            for result in results:
                for pre_token in result.keys():
                    if pre_token in self.pre_tokens:
                        self.pre_tokens[pre_token].count += result[pre_token]
                    else:
                        byte_pairs = deque()
                        for byte in pre_token.encode("utf-8"):
                            byte_pairs.append(bytes([byte]))
                        self.pre_tokens[pre_token] = Pretoken(result[pre_token], byte_pairs)

        self.pre_token_list = list(self.pre_tokens.values())

    def init_bytepair(self):
        for token_key, token_value in enumerate(self.pre_token_list):
            count = token_value.count
            byte_pairs = token_value.byte_pairs

            last_token = None
            new_byte_pairs = deque()
            
            while len(byte_pairs):
                token = byte_pairs.popleft()
                new_byte_pairs.append(token)
                if last_token:
                    if (last_token, token) in self.byte_pairs.keys():
                        byte_pair = self.byte_pairs[(last_token, token)]
                        byte_pair.frequency += count
                        byte_pair.pretoken.append(token_key)
                        
                        
                    else:
                        byte_pair = BytePair(last_token, token, count, [token_key])
                        self.byte_pairs[(last_token, token)] = byte_pair

                last_token = token
                
            token_value.byte_pairs = new_byte_pairs
        
        for key, value in self.byte_pairs.items():
            self.byte_pairs_heap.push(key, value)
        
    def get_max_byte_pair(self):
        #return max(self.byte_pairs.values())
        return self.byte_pairs_heap.peek()[1]

    def merge_bytepair(self):
        old_byte_pair = self.get_max_byte_pair()
        first = old_byte_pair.first 
        second = old_byte_pair.second
        self.merges.append((bytes(first),bytes(second)))
        self.vocab[len(self.vocab)] = bytes(first + second)

        need_update_keys = set()
        for token_key in old_byte_pair.pretoken:
            token_value = self.pre_token_list[token_key]
            count = token_value.count
            byte_pairs = token_value.byte_pairs
            
            last_token = None
            new_byte_pairs = deque()
            while len(byte_pairs):
                token = byte_pairs.popleft()
                if last_token and last_token == first and token == second:
                    new_byte_pairs.pop()
                    token = first + second

                    key = (first, second)
                    byte_pair = self.byte_pairs[key]
                    byte_pair.frequency -= count
                    if byte_pair.frequency <= 0:
                        del self.byte_pairs[key]
                    need_update_keys.add(key)
                    self.byte_pairs_heap.delete(key)
  
                    if len(new_byte_pairs):
                        
                        key = (new_byte_pairs[-1], first)
                        byte_pair = self.byte_pairs[key]
                        byte_pair.frequency -= count
                        if byte_pair.frequency <= 0:
                            del self.byte_pairs[key]
                        need_update_keys.add(key)
                        self.byte_pairs_heap.delete(key)  
                        
                        key = (new_byte_pairs[-1], token)
                        if key in self.byte_pairs.keys():
                            byte_pair = self.byte_pairs[key]
                            byte_pair.frequency += count
                            byte_pair.pretoken.append(token_key)
                        else:
                            byte_pair = BytePair(new_byte_pairs[-1], token, count, [token_key])
                            self.byte_pairs[key] = byte_pair
                        
                        need_update_keys.add(key)
                        self.byte_pairs_heap.delete(key)

                    if len(byte_pairs):
                        
                        key = (second, byte_pairs[0])
                        byte_pair = self.byte_pairs[key]
                        byte_pair.frequency -= count
                        if byte_pair.frequency <= 0:
                            del self.byte_pairs[key]
                        need_update_keys.add(key)
                        self.byte_pairs_heap.delete(key)
                        
                        key = (token, byte_pairs[0])
                        if key in self.byte_pairs.keys():
                            byte_pair = self.byte_pairs[key]
                            byte_pair.frequency += count
                            byte_pair.pretoken.append(token_key)
                        else:
                            byte_pair = BytePair(token, byte_pairs[0], count, [token_key])
                            self.byte_pairs[key] = byte_pair
                        need_update_keys.add(key)
                        self.byte_pairs_heap.delete(key)
                new_byte_pairs.append(token)
                last_token = token
            token_value.byte_pairs = new_byte_pairs
        
        for key in need_update_keys:
            if key in self.byte_pairs:
                self.byte_pairs_heap.push(key, self.byte_pairs[key])
        
    def train(self):
        with open(self.file_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()

            chunk_num = file_size // self.chunk_size + 1
            boundaries = find_chunk_boundaries(f, chunk_num, b"<|endoftext|>")
            
            tasks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                # f.seek(start)
                # chunk = f.read(end - start).decode("utf-8", errors="ignore")
                tasks.append((self.file_path, start, end))

            self.pre_tokenation(tasks)
            print("Initializing byte pair frequencies...")
            self.init_bytepair()
            print("Start to merge byte pairs")
            for i in tqdm(range(self.vocab_size - len(self.vocab)), desc="Merging byte pairs"):
                self.merge_bytepair()
    
        return self.vocab, self.merges

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None, chunk_size = int(1 * 1024 ** 2), num_processes=16):
        """
        从给定的词汇表、合并规则列表和（可选的）特殊token列表构造分词器
        """
        self.vocab = vocab
        self.vocab_to_index = {v: k for k, v in vocab.items()} if isinstance(vocab, dict) else vocab
        self.merges = merges
        self.merge_to_index = {merge:i for i,merge in enumerate(self.merges)}
        self.chunk_size = chunk_size
        self.num_processes = num_processes
        # 缓存已经编码的pre_token
        self.pre_tokens = {}
        self.pre_special_tokens = {}


        if isinstance(special_tokens, list):
            if "<|endoftext|>" not in special_tokens:
                special_tokens.append("<|endoftext|>")
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            for special_token in special_tokens:
                if special_token.encode('utf-8') not in self.vocab_to_index:
                    self.vocab_to_index[special_token.encode('utf-8')] = len(self.vocab)
                    self.vocab[len(self.vocab)] = special_token.encode('utf-8')
        self.special_tokens = special_tokens
        if self.special_tokens:
            self.concat_special_tokens = re.compile("|".join([re.escape(special_token) for special_token in self.special_tokens]))
        
            for special_token in self.special_tokens:
                self.pre_special_tokens[special_token.encode('utf-8')] = [self.vocab_to_index[special_token.encode('utf-8')]]
        
        self.pre_tokens.update(self.pre_special_tokens)
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.pool = Pool(processes=self.num_processes)

        
    def __del__(self):
        self.pool.close()
        self.pool.join()

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        """
        类方法：从序列化的词表文件、merges文件构造并返回分词器
        （文件格式与BPE训练代码的输出格式一致）
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_serializable = json.load(f)

        # 还原格式：dict[int, bytes]
        vocab: dict[int, bytes] = {}
        for str_idx, hex_token in vocab_serializable.items():
            idx = int(str_idx)                # 字符串索引 → 整数索引
            token = bytes.fromhex(hex_token)  # 十六进制字符串 → 原始bytes
            vocab[idx] = token
        
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 跳过注释行、空行
                if not line or line.startswith("#"):
                    continue

                # 按空格分割两个十六进制串 → 还原为 bytes 对
                hex_a, hex_b = line.split()
                a = bytes.fromhex(hex_a)
                b = bytes.fromhex(hex_b)
                merges.append((a, b))
        
        tokenizer = cls(vocab, merges, special_tokens)
        
        return tokenizer
    
    def pretokenize(self, text):
        """
        核心预分词方法，将text预分词并返回待合并的token ids：[[1,2],[8],...]
        """
        
        token_seqs = []

        
        special_tokens_seq = []
        if self.special_tokens:
            splited_chunk = re.split(self.concat_special_tokens, text,)
            matches = re.finditer(self.concat_special_tokens, text)
            special_tokens_seq = [match.group().encode("utf-8") for match in matches]
        else:
            splited_chunk = [text]
        

        for i, chunk in enumerate(splited_chunk):
            # pre tokenization    
            pre_token_iter = re.finditer(self.PAT, chunk)
            for pre_token in pre_token_iter:
                pre_token = pre_token.group()
                token_seqs.append(pre_token)
                if pre_token not in self.pre_tokens:
                    #pre_token_result =[bytes([byte]) for byte in pre_token.encode("utf-8")]
                    # 表示未解码
                    self.pre_tokens[pre_token] = []

            if i < len(special_tokens_seq):
                token_seqs.append(special_tokens_seq[i])
                

        return token_seqs

    @staticmethod
    def merge_tokens(tasks):
        pre_tokens = {}
        
        def _merge(pre_token_bytes_seq, merges):
            byte_pairs = [
                (pre_token_bytes_seq[i], pre_token_bytes_seq[i+1]) 
                for i in range(len(pre_token_bytes_seq) - 1)
            ]
            bps_visited = set(byte_pairs)
            bps = [(merges[byte_pair],byte_pair) for byte_pair in byte_pairs if byte_pair in merges]
            
            heapq.heapify(bps)
            tokens = deque(pre_token_bytes_seq)
            while len(bps):
                
                idx, merge = heapq.heappop(bps)
                
                new_tokens = deque()
                first, second = merge
                
                while tokens:
                    current = tokens.popleft()
                    
                    if tokens and current == first and tokens[0] == second:
                        merged = first + second  
                        
                        tokens.popleft()
                        
                        if len(new_tokens):
                            if (new_tokens[-1],merged) in merges.keys() and (new_tokens[-1],merged) not in bps_visited:
                                bps_visited.add((new_tokens[-1],merged))
                                heapq.heappush(bps, (merges[(new_tokens[-1],merged)], (new_tokens[-1],merged)))

                        if len(tokens):
                            if (merged, tokens[0]) in merges.keys() and (merged, tokens[0]) not in bps_visited:
                                bps_visited.add((merged, tokens[0]))
                                heapq.heappush(bps, (merges[(merged, tokens[0])], (merged, tokens[0])))

                        new_tokens.append(merged)
                    else:
                        new_tokens.append(current)
                
                tokens = new_tokens  # 更新 tokens 为结果
            pre_token_bytes_seq = list(tokens)
            return pre_token_bytes_seq

        for task in tasks:
            pre_token = task[0]
            merges = task[1]

            pre_token_bytes_seq = [bytes([byte]) for byte in pre_token.encode("utf-8")]

            pre_token_bytes_seq = _merge(pre_token_bytes_seq, merges)
            pre_tokens[pre_token] = pre_token_bytes_seq
        
        return pre_tokens

    def encode(self, text: str) -> list[int]:
        """将输入文本编码为token ID序列"""
        token_seqs = self.pretokenize(text)
        token_ids = []

        tasks = [(k, self.merge_to_index) for k,v in self.pre_tokens.items() if not v]

        def split_into_chunks(tasks, num_chunks):
            """将任务列表分成多个子任务块"""
            chunk_size = max(1, len(tasks) // num_chunks)
            chunks = [tasks[i:i+chunk_size] for i in range(0, len(tasks), chunk_size)]
            return chunks
        
        
        if False and len(tasks) > 2000:
            # 此任务被证明为负优化
            task_chunks = split_into_chunks(tasks, self.num_processes)
            for pre_tokens in self.pool.imap(self.merge_tokens, task_chunks):
                for pre_token, pre_token_bytes_seq in pre_tokens.items():
                    self.pre_tokens[pre_token] = [self.vocab_to_index[bs] for bs in pre_token_bytes_seq]

        else:
            pre_tokens = self.merge_tokens(tasks)
            for pre_token, pre_token_bytes_seq in pre_tokens.items():
                self.pre_tokens[pre_token] = [self.vocab_to_index[bs] for bs in pre_token_bytes_seq]

        token_ids = [
            tid
            for token_seq in token_seqs
            for tid in self.pre_tokens[token_seq]
        ]
        
        if len(self.pre_tokens) > 1000000:
            keep_count = int(len(self.pre_tokens) * 0.9)
            keys = list(self.pre_tokens.keys())
            keys_to_keep = random.sample(keys, keep_count)

            self.pre_tokens = {k: self.pre_tokens[k] for k in keys_to_keep}
            self.pre_tokens.update(self.pre_special_tokens) 
        
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        对字符串可迭代对象（如Python中的文件对象）惰性编码，返回token ID生成器
        用于大文件的内存高效分词，避免直接加载整个文件到内存
        """
        
        chunk_start_pos = 0
        pre_chunk = None
        while True:
            if not pre_chunk:
                chunk_start_pos = iterable.tell()
            chunk = iterable.read(self.chunk_size)

            if not chunk:
                break
            
            is_last_chunk = len(chunk) < self.chunk_size

            if pre_chunk:
                chunk = pre_chunk + chunk

            pre_chunk = chunk

            if not is_last_chunk:
                if self.special_tokens:
                    # 通过special_tokens截断
                    #concat_special_tokens = "|".join([re.escape(special_token) for special_token in self.special_tokens])
                    matches = list(re.finditer(self.concat_special_tokens, chunk))
                    if len(matches) > 0:
                        chunk = chunk[:matches[-1].span()[1]]
                    else:
                        chunk = chunk[:-len(max(self.special_tokens, key=len))]
                        pretokens = re.findall(self.PAT, chunk)
                        if pretokens:
                            endtoken_len = len(pretokens[-1])
                            chunk = chunk[:-endtoken_len]
                else:
                    pretokens = re.findall(self.PAT, chunk)
                    if pretokens:
                        endtoken_len = len(pretokens[-1])
                        chunk = chunk[:-endtoken_len]
            
                # 如果chunk的长度为0，那么整个chunk是一个很长的pretoken，增加长度避免pretoken截断    
                if not chunk and len(pre_chunk) < 4 * self.chunk_size:
                    continue
                else:
                    chunk = pre_chunk
            
            pre_chunk = None

            chunk_len = len(chunk)
            if chunk_len == 0:
                break
            
            iterable.seek(chunk_start_pos)
            iterable.read(chunk_len)

            token_ids = self.encode(chunk)

            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """将token ID序列解码为文本"""
        tokens = [self.vocab[_id] for _id in ids]
        full_bytes = b''.join(tokens)
        full_text = full_bytes.decode('utf-8', errors='replace')

        return full_text

