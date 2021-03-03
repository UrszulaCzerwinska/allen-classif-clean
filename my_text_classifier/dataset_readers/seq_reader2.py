import itertools
import json
from typing import Dict, Iterable, List
from overrides import overrides

import numpy as np

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data import TokenIndexer, Tokenizer
from allennlp.data.instance import Instance
from allennlp.data.fields.field import Field
from allennlp.data.fields import TextField, LabelField, ListField, ArrayField, MultiLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.tokenizers.token_class import Token


@DatasetReader.register("seq-classification-reader-2")
class SeqClassificationReader(DatasetReader):
    """
    Reads a file from Pubmed-RCT dataset. Each instance contains an abstract_id, 
    a list of sentences and a list of labels (one per sentence).
    Input File Format: Example abstract below:
        {
        "abstract_id": 5337700, 
        "sentences": ["this is motivation", "this is method", "this is conclusion"], 
        "labels": ["BACKGROUND", "RESULTS", "CONCLUSIONS"]
        }
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 sent_max_len: int = 100,
                 max_sent_per_example: int = 20,
                 use_sep: bool = True,
                 sci_sum: bool = False,
                 use_abstract_scores: bool = True,
                 sci_sum_fake_scores: bool = True,
                 predict: bool = False,
                 **kwargs
                 ) -> None:
        super().__init__(manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,  **kwargs)
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.sent_max_len = sent_max_len
        self.use_sep = use_sep
        self.predict = predict
        self.sci_sum = sci_sum
        self.max_sent_per_example = max_sent_per_example
        self.use_abstract_scores = use_abstract_scores
        self.sci_sum_fake_scores = sci_sum_fake_scores,

    def read_one_example(self, json_dict):
        
        sentences = json_dict["sentences"]
        labels = json_dict["labels"]
        
        return (sentences, labels)
  

    def text_to_instance(self,
                         sentences: List[str], labels: List[str] = None,
                         ) -> Instance:
        if not self.predict:
            assert len(sentences) == len(labels)
     
        if self.use_sep:
            tokenized_sentences = [self._tokenizer.tokenize(s)[:self.sent_max_len] + [Token("[SEP]")] for s in sentences]
            sentences = [list(itertools.chain.from_iterable(tokenized_sentences))[:-1]]
        else:
            # Tokenize the sentences
            sentences = [
                self._tokenizer.tokenize(sentence_text)[:self.sent_max_len]
                for sentence_text in sentences
            ]

        fields: Dict[str, Field] = {}
        fields["sentences"] = ListField([
                TextField(sentence)
                for sentence in sentences
        ])

        if labels is not None:
            if isinstance(labels[0], list):
                fields["labels"] = ListField([
                        MultiLabelField(label) for label in labels
                    ])
            else:
                # make the labels strings for easier identification of the neutral label
                # probably not strictly necessary
                if self.sci_sum:
                    fields["labels"] = ArrayField(np.array(labels))
                else:
                    fields["labels"] = ListField([
                            LabelField(str(label)+"_label") for label in labels
                        ])
        return Instance(fields)
        

    def apply_token_indexers(self, instance: Instance) -> None:
        for text_field in instance["sentences"].field_list:
            text_field.token_indexers = self._token_indexers

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as f:
            for line in self.shard_iterable(f):
                json_dict = json.loads(line)
                sentences, labels = self.read_one_example(json_dict)
                yield self.text_to_instance(
                sentences=sentences,
                labels=labels,
                )
                        

