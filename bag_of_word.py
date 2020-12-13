from transformers import BertTokenizer 



if __name__ == "__main__":
    tokenizer = BertTokenizer('data/vocab.txt')
    s = 'a cat shsiss dog'
    s = tokenizer.tokenize(s)
    print(tokenizer.convert_tokens_to_ids(s))


