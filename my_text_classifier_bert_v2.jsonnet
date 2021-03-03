local bert_model = "bert-base-uncased";

{
    "dataset_reader" : {
        "type": "SeqClassificationReader",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
            }
        },
        "sent_max_len": 10,
        "max_sent_per_example": 80,
        "use_sep": 1,
        "sci_sum": 0,
        "use_abstract_scores": 0,
        "sci_sum_fake_scores": 0,
    },
    "train_data_path": "data_seq/CSAbstruct/train.jsonl",
    "validation_data_path": "data_seq/CSAbstruct/dev.jsonl",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model
                }
            }
        },
        "encoder": {
            "type": "bert_pooler",
            "pretrained_model": bert_model,
            "requires_grad": true
        }
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        },
        "num_epochs": 5
    }
}
