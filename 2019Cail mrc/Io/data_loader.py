import torch
import config.args as args
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer
from preprocessing.data_processor import read_qa_examples, convert_examples_to_features
from util.Logginger import init_logger

logger = init_logger("bert_class", logging_path=args.log_path)


def init_params():
    tokenizer = BertTokenizer(vocab_file=args.VOCAB_FILE)
    return tokenizer


def create_batch_iter(mode):
    """构造迭代器"""
    tokenizer = init_params()
    if mode == "train":
        examples = read_qa_examples(args.data_dir, "train")
        batch_size = args.train_batch_size
    elif mode == "dev":
        examples = read_qa_examples(args.data_dir, "dev")
        batch_size = args.eval_batch_size
    else:
        raise ValueError("Invalid mode %s" % mode)

    # 特征
    features = convert_examples_to_features(examples,
                                            tokenizer,
                                            args.max_seq_length,
                                            args.doc_stride,
                                            args.max_query_length,
                                            is_training=True)

    logger.info("  Num Features = %d", len(features))
    logger.info("  Batch size = %d", batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
    end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
    answer_types = torch.tensor([f.answer_type for f in features], dtype=torch.long)

    # 数据集
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, start_positions, end_positions, answer_types)

    if mode == "train":
        num_train_steps = int(
            len(features) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        batch_size = args.train_batch_size

        logger.info("  Num steps = %d", num_train_steps)
        if args.local_rank == -1:
            sampler = RandomSampler(data)
        else:
            sampler = DistributedSampler(data)
    elif mode == "dev":
        sampler = SequentialSampler(data)
    else:
        raise ValueError("Invalid mode %s" % mode)

    # 迭代器
    iterator = DataLoader(data, sampler=sampler, batch_size=batch_size)

    if mode == "train":
        return iterator, num_train_steps
    elif mode == "dev":
        return iterator
    else:
        raise ValueError("Invalid mode %s" % mode)


