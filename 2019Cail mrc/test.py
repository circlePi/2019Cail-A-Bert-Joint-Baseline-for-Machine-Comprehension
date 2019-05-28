from Io.data_loader import create_batch_iter
from preprocessing.data_processor import read_squad_data, convert_examples_to_features, read_qa_examples
from pytorch_pretrained_bert.tokenization import BertTokenizer
from predict.predict import main

if __name__ == "__main__":
    read_squad_data("data/small_train_data.json", "data/")
    # examples = read_qa_examples("data/", "train")
    # print(len(examples))
    # features = convert_examples_to_features(examples,
    #                              tokenizer=BertTokenizer("pretrained_model/vocab.txt"),
    #                              max_seq_length=512,
    #                              doc_stride=500,
    #                              max_query_length=32,
    #                              is_training=True)
    # print(len(features))
    # main()