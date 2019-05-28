import os
import json
import random
import collections
from tqdm import tqdm
import config.args as args
from util.Logginger import init_logger
from pytorch_pretrained_bert.tokenization import BertTokenizer

logger = init_logger("QA", logging_path=args.log_path)


class InputExample(object):
    "Template for a single data"
    def __init__(self,
                 qas_id,                    # question id
                 question_text,             # question text
                 doc_tokens,                # context
                 orig_answer_text=None,     # answer text
                 start_position=None,       # For Yes, No & no-answer, start_position = 0
                 end_position=None,         # For Yes, No & no-answer, start_position = 0
                 answer_type=None           # We denote answer type as Yes: 0 No: 1 no-answer: 2 long-answer: 3
                 ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.answer_type = answer_type


class InputFeatures(object):
    "Feature to feed into model"
    def __init__(self,
                 unique_id,                   # feature id
                 example_index,               # example index, note this is different from qas_id
                 doc_span_index,              # split context index
                 tokens,                      # question token + context + flag character
                 token_to_orig_map,           # token index before BertTokenize
                 token_is_max_context,
                 input_ids,                   # model input, the id of tokens
                 input_mask,
                 segment_ids,                 # For distinguishing question & context
                 start_position=None,
                 end_position=None,
                 answer_type=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.answer_type = answer_type


def train_val_split(X, y, valid_size=0.25, random_state=2019, shuffle=True):
    """
    训练集验证集分割
    :param X: sentences
    :param y: labels
    :param random_state: 随机种子
    """
    logger.info('train val split')

    train, valid = [], []
    bucket = [[] for _ in [i for i in range(len(args.answer_type))]]

    for data_x, data_y in tqdm(zip(X, y), desc='bucket'):
        bucket[int(data_y)].append((data_x, data_y))

    del X, y

    for bt in tqdm(bucket, desc='split'):
        N = len(bt)
        if N == 0:
            continue
        test_size = int(N * valid_size)

        if shuffle:
            random.seed(random_state)
            random.shuffle(bt)

        valid.extend(bt[:test_size])
        train.extend(bt[test_size:])

    if shuffle:
        random.seed(random_state)
        random.shuffle(valid)
        random.shuffle(train)

    return train, valid


def read_squad_data(raw_data, save_dir, is_training=True):
    logger.info("Read raw squad data...")
    logger.info("train_dev_split is %s" % str(is_training))
    logger.info("test data path is %s" % raw_data)
    with open(raw_data, "r", encoding="utf-8") as fr:
        data = json.load(fr)
        data = data["data"]
    samples = []
    for e in data:
        case_id = e["caseid"]
        domain = e["domain"]
        paragraphs = e["paragraphs"]
        # For small train, we just observed one paragraph in the paragraph list
        for paragraph in paragraphs:
            case_name = paragraph["casename"]
            context = paragraph["context"]
            qas = paragraph["qas"]
            for qa in qas:
                question = qa["question"]
                answers = qa["answers"]
                example_id = qa["id"]
                assert len(answers) <= 1, "Found more than one answer for one question"

                is_impossible = qa["is_impossible"]    # if true, means long-answer, Yes & no, otherwise, means no-answer
                # no-answer
                if is_impossible == "true" or len(answers) == 0:
                    answer_type = "no-answer"
                    answer_text = "unknown"
                    start_position = 0
                    end_position = 0
                    sample = {"case_id": case_id, "context": context, "domain": domain, "case_name": case_name,
                              "question": question, "answer_type": answer_type, "answer_text": answer_text,
                              "start_position": start_position, "end_position": end_position, "example_id": example_id}
                    samples.append(sample)
                else:
                    for answer in answers:
                        start_position = answer["answer_start"]
                        answer_text = answer["text"]
                        # For Yes & No
                        if start_position == -1:
                            answer_type = answer["text"]
                            start_position = 0
                            end_position = 0
                        # For long-answer
                        else:
                            start_position = answer["answer_start"]
                            end_position = start_position + len(answer_text) -1
                            answer_type = "long-answer"
                        sample = {"case_id": case_id, "context": context, "domain": domain, "case_name": case_name,
                                  "question": question, "answer_type": answer_type, "answer_text": answer_text,
                                  "start_position": start_position, "end_position": end_position, "example_id": example_id}
                        samples.append(sample)
    if is_training:
        y = [args.answer_type[sample["answer_type"]] for sample in samples]
        train, valid = train_val_split(samples, y)
        logger.info("Train set size is %d" % len(train))
        logger.info("Dev set size is %d" % len(valid))
        with open(os.path.join(save_dir, "train.json"), 'w') as fr:
            for t in train:
                print(json.dumps(t[0], ensure_ascii=False), file=fr)
        with open(os.path.join(save_dir, "dev.json"), 'w') as fr:
            for v in valid:
                print(json.dumps(v[0], ensure_ascii=False), file=fr)
    else:
        with open(os.path.join(save_dir, "test.json"), 'w') as fr:
            logger.info("Test set size is %d" %len(samples))
            for sample in samples:
                print(json.dumps(sample), file=fr)


def read_qa_examples(data_dir, corpus_type):
    assert corpus_type in ["train", "dev", "test"], "Unknown corpus type"
    examples = []
    with open(os.path.join(data_dir, corpus_type +'.json'), 'r') as fr:
        for i, data in enumerate(fr):
            data = json.loads(data.strip("\n"))
            example = InputExample(qas_id=data["example_id"],
                                   question_text=data["question"],
                                   doc_tokens=data["context"],
                                   orig_answer_text=data["answer_text"],
                                   start_position=data["start_position"],
                                   end_position=data["end_position"],
                                   answer_type=data["answer_type"])
            
            examples.append(example)
                
    return examples


def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_seq_length,
                                 doc_stride,
                                 max_query_length,
                                 is_training):
    unique_id = 10000000

    features = []
    for example_index, example in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None

        if is_training:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                           split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            answer_type = None
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                if example.answer_type != "no-answer":
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                        answer_type = "no-answer"
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                        answer_type = example.answer_type
                else:
                    start_position = 0
                    end_position = 0
                    answer_type = "no-answer"

                answer_type = args.answer_type[answer_type]

            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                        "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                        "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                            "answer: %s" % (answer_text))
                    logger.info("answer_type: %s" %answer_type)

                

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    answer_type=answer_type))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


if __name__ == '__main__':
    read_squad_data("data/small_train_data.json", "data/")
    examples = read_qa_examples("data/", "train")
    convert_examples_to_features(examples,
                                 tokenizer=BertTokenizer("pretrained_model/vocab.txt"),
                                 max_seq_length=512,
                                 doc_stride=500,
                                 max_query_length=32,
                                 is_training=True)
