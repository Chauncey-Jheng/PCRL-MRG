import json

error_sample_words = ["胸","肺","腹","肋","心影","心包","心脏","冠状","冠脉","椎","肝","胆","左肾","右肾",
                      "胰","腋","纵隔","甲状腺","腹腔","盆腔","下腔静脉","骨盆","膀胱","子宫",
                      "臀","胃","脾","腕","桡","尺骨","关节","耻骨","股骨","前列腺","主动脉","动脉壁",
                      "食管","肠","尿","气管","胫","排骨","锁骨","片絮状","肱骨","粥样","乳腺","颈部",
                      "片中所示","彩超","超声","穿刺",
                      "颌骨CT","颌下腺","面部肌肉"]

layer_entities ={'颅底层面眦耳线层面': ['上颌窦', '筛窦', '蝶窦', '第四脑室', '脑干', '小脑', '小脑半球',],
    '鞍上池层面': ['额叶', '颞叶', '颞枕叶', '枕叶', '中脑'],
    '第三脑室下部层面': ['额叶', '丘脑', '侧脑室', '枕叶', '第三脑室', '小脑幕', '颞叶', '中线'],
    '第三脑室上部层面': ['基底节区', '大脑镰', '侧脑室', '枕叶', '颞枕叶', '丘脑', '颞叶', '第三脑室', '额叶', '中线'],
    '侧脑室体部层面': ['侧脑室', '额叶', '放射冠', '颞叶', '枕叶', '颞枕叶', '大脑镰', '中线'],
    '侧脑室上部层面': ['侧脑室', '大脑镰', '额叶', '顶叶', '脑沟', '脑室', '枕叶', '中线'],
    '半卵圆中心层面': ['额叶', '半卵圆中心', '顶叶', '大脑镰', '脑沟', '脑室', '脑实质', '中线'],
    '大脑皮质上部层面': ['大脑镰', '脑回', '顶叶', '放射冠', '额叶', '脑沟']}

entities = []
for value in layer_entities.values():
    entities += value
entities = set(entities)
print(entities)
print(len(entities))
pathology_entity_Chinese = list(entities)

import re
def remove_sentences_with_keywords(text, keywords):
    text = text.replace("，建议","。建议")
    text = text.replace("。建议进一步检查。","。")

    text = text.replace("脑千","脑干")
    text = text.replace("题","颞")
    text = text.replace("颢","颞")
    text = text.replace("若侧","右侧")
    text = text.replace("脑外伤复查示：","")

    sentences = re.split(r'(。|；)', text)
    filtered_sentences = []
    for sentence in sentences:
        if any(keyword in sentence for keyword in keywords):
            continue
        else:
            filtered_sentences.append(sentence)
    text =  "".join(filtered_sentences)
    text = re.sub(r'[。]+', '。', text)
    text = re.sub(r'[；]+', '；', text)
    text = re.sub(r'；。','。', text)
    text = re.sub(r'。；','；', text)
    text = re.sub(r'[；]+', '；', text)
    text = re.sub(r'[。]+', '。', text)    
    text = re.sub(r'^[；。]+', '', text)

    text = text.replace("头颅CT平扫：","")
    text = text.replace("头颅平扫：","")

    return text

def generate_entities_from_text(text, keywords):
    sample = {
        "entities":[],
        "entities_lession":[]
    }
    sentences = re.split(r'(。|；|，)', text)
    for sentence in sentences:
        for keyword in keywords:
            if keyword in sentence:
                continue_flag = False
                for k in keywords:
                    if keyword in k and keyword != k and k in sentence:
                        continue_flag = True
                        break
                if continue_flag:
                    continue
                if keyword in sample["entities"]:
                    sample["entities_lession"][sample["entities"].index(keyword)] += "；" + sentence
                    continue
                sample["entities"].append(keyword)
                sample['entities_lession'].append(sentence)
        else:
            continue
    return sample["entities"], sample["entities_lession"]

with open('CTRG_dataset/origin_samples.json', 'r') as file:
    data = json.load(file)

with open('/home/bjutcv/data/zcx/dataset/CTRG-Brain/feature/total_image_list_final.json', 'r') as file:
    mapping_id_with_24_images = json.load(file)

filtered_data = []

for sample in data:
    sample["impression"] = remove_sentences_with_keywords(sample["impression"], error_sample_words)
    sample["findings"] = remove_sentences_with_keywords(sample["findings"], error_sample_words)
    sample["images"] = mapping_id_with_24_images[sample["id"]]
    sample["finding_entities"], sample["finding_etities_lession"] = generate_entities_from_text(sample["findings"], pathology_entity_Chinese)
    sample["impression_entities"], sample["impression_etities_lession"] = generate_entities_from_text(sample["impression"], pathology_entity_Chinese)
    filtered_data.append(sample)

with open('CTRG_dataset/samples_filtered.json', 'w', encoding='utf-8') as json_file:
    json.dump(filtered_data, json_file, ensure_ascii=False, indent=4)


total_length = len(filtered_data)
len1 = total_length * 7 // 10
len2 = total_length * 2 // 10
len3 = total_length - len1 - len2
print(len1,len2,len3)

train_data = filtered_data[:len1]
test_data = filtered_data[len1:len1+len2]
val_data = filtered_data[len1+len2:]

with open('CTRG_dataset/train.json', 'w',encoding="utf-8") as train_file:
    json.dump(train_data, train_file, ensure_ascii=False, indent=4)

with open('CTRG_dataset/test.json', 'w',encoding="utf-8") as test_file:
    json.dump(test_data, test_file, ensure_ascii=False, indent=4)

with open('CTRG_dataset/validation.json', 'w',encoding="utf-8") as val_file:
    json.dump(val_data, val_file, ensure_ascii=False, indent=4)
