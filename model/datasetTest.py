from datasets import load_dataset

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

ds_jatzingueni = load_dataset('BAPE119/jatzingueni-purepecha-espaniol', split='train')
ds_opus = load_dataset(f"opus_books", f"en-it", split='train')

sentences = get_all_sentences(ds_jatzingueni, "es")
for sentence in sentences:
    print(sentence)