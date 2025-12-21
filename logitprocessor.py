from transformers.generation import LogitsProcessor
import torch
from typing import Callable, List, Optional, Dict


from typing import Dict, List, Optional


class TrieNode:
    __slots__ = ("children", "is_end")

    def __init__(self):
        # token_id -> TrieNode
        self.children: Dict[int, "TrieNode"] = {}
        self.is_end: bool = False


class PrefixTrie:
    def __init__(self):
        self.root = TrieNode()
        self.max_depth = 0


    def insert(self, seq: List[int]):
        node = self.root
        depth = 0

        for token in seq:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
            depth += 1

        node.is_end = True
        self.max_depth = max(self.max_depth, depth)

    def bulk_insert(self, seqs: List[List[int]]):
        for seq in seqs:
            self.insert(seq)


    def get_next_tokens(self, prefix: List[int]) -> Optional[List[int]]:
        node = self.root

        for token in prefix:
            if token not in node.children:
                return None
            node = node.children[token]

        if not node.children:
            return [] 

        return list(node.children.keys())

    def is_valid_prefix(self, prefix: List[int]) -> bool:
        node = self.root
        for token in prefix:
            if token not in node.children:
                return False
            node = node.children[token]
        return True

    def is_complete(self, seq: List[int]) -> bool:
        node = self.root
        for token in seq:
            if token not in node.children:
                return False
            node = node.children[token]
        return node.is_end


    def __contains__(self, seq: List[int]) -> bool:
        return self.is_complete(seq)

    def __len__(self) -> int:
        return self._count_leaves(self.root)

    def _count_leaves(self, node: TrieNode) -> int:
        cnt = 1 if node.is_end else 0
        for child in node.children.values():
            cnt += self._count_leaves(child)
        return cnt



class FSM:
    def __init__(self, start_state=0):
        self.start_state = start_state
        self.states: Dict[int, Dict[int, int]] = {}

    def add_transition(self, state: int, token: int, next_state: int):
        if state not in self.states:
            self.states[state] = {}
        self.states[state][token] = next_state

    def next_tokens(self, state: int) -> List[int]:
        return list(self.states.get(state, {}).keys())

    def next_state(self, state: int, token: int) -> Optional[int]:
        return self.states.get(state, {}).get(token, None)


class ConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        prefix_allowed_tokens_fn: Callable[[int, List[int]], List[int]],
        num_beams: int,
        prefix_length: int = 3,          
        use_trie: bool = False,
        trie: Optional[PrefixTrie] = None,
        use_fsm: bool = False,
        fsm: Optional[FSM] = None
    ):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

        self.prefix_length = prefix_length  
        self.count = 0

        self.use_trie = use_trie
        self.trie = trie

        self.use_fsm = use_fsm
        self.fsm = fsm


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        batch_beam = scores.shape[0]
        batch = batch_beam // self._num_beams

        seq_len = input_ids.shape[-1]
        input_ids = input_ids.view(batch, self._num_beams, seq_len)

        allowed_indices = []
        allowed_tokens = []

        for batch_id in range(batch):
            for beam_id in range(self._num_beams):

                if self.count == 0:
                    prefix = input_ids[batch_id, beam_id][-self.prefix_length:].tolist()
                else:
                    prefix = input_ids[batch_id, beam_id][-self.count:].tolist()

                flat_index = batch_id * self._num_beams + beam_id

                if self.use_trie and self.trie is not None:
                    allowed = self.trie.get_next_tokens(prefix)
                    if allowed:
                        allowed_indices.extend([flat_index] * len(allowed))
                        allowed_tokens.extend(allowed)
                    continue 

                if self.use_fsm and self.fsm is not None:
                    allowed = self.fsm.next_tokens(state=0)  
                    allowed_indices.extend([flat_index] * len(allowed))
                    allowed_tokens.extend(allowed)
                    continue

                allowed = self._prefix_allowed_tokens_fn(batch_id, prefix)
                if len(allowed) > 0:
                    allowed_indices.extend([flat_index] * len(allowed))
                    allowed_tokens.extend(allowed)

        mask = torch.full_like(scores, float('-inf'))
        if len(allowed_indices) > 0:
            mask[allowed_indices, allowed_tokens] = 0.0

        scores = scores + mask
        self.count += 1

        return scores
