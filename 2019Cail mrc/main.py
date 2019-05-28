import json
from predict.predict import main


if __name__ == '__main__':
    test_data_path = "../data/data.json"
    data_dir = "../result/"
    main(test_data_path, data_dir)
    result = []
    with open("../result/nbest_predictions.json", "r", encoding="utf-8") as fr:
        data = json.load(fr)
    for key, value in data.items():
        res = {"answer": value[0]["text"], "id": key}
        result.append(res)
    with open("../result/result.json", 'w') as fr:
        json.dump(result, fr)
