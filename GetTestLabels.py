import json

with open('data\\emo-test.json', encoding='utf-8') as labels_file, open('data\\testwithoutlabels.txt', 'r', encoding='utf-8') as inp, open('data\\test.txt', 'w', encoding='utf-8') as outp:
    labels = json.load(labels_file)
    outp.write(inp.readline()[:-1] + '\tlabel\n')
    labels = labels['Label']
    for i in range(5509):
        outp.write(f'{inp.readline().rstrip()}\t{labels[str(i)]}\n')
