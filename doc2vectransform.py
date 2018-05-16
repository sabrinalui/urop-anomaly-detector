import gensim
import ipdb
import os
import numpy

def gen_doc2vecmodel():

    # Set file names for train and test data
    test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
    lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
    lee_test_file = test_data_dir + os.sep + 'lee.cor'


    def read_corpus(fname, tokens_only=False):
        with open(fname, encoding="iso-8859-1") as f:
            for i, line in enumerate(f):
                if tokens_only:
                    yield gensim.utils.simple_preprocess(line)
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

    train_corpus = list(read_corpus(lee_train_file))
    test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

    model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=55)
    model.build_vocab(train_corpus)

    print("Training doc2vec model...")
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    print("Training complete.")

    return model

def doc2vectorizer(tweets):
    model = gen_doc2vecmodel()
    return numpy.array([model.infer_vector(tweet) for tweet in tweets])