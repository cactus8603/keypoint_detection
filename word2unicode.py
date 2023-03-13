import pickle

def unicode2word(raw_str_unicode):
    str_unicode = '\\u' + raw_str_unicode
    byte_unicode = str_unicode.encode('utf-8')
    word = byte_unicode.decode('unicode_escape')
    return word

with open('./word_list/unicode2onehot.pkl', 'rb') as f:
    uni2word = pickle.load(f)

for j, unicode in enumerate(uni2word.keys()):
    uni2word[unicode] = unicode2word(unicode)
print(uni2word)

with open('./word_list/unicode2chara.pkl', 'wb') as f:
    pickle.dump(uni2word, f)
