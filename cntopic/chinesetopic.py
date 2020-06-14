import gensim
from gensim.corpora import Dictionary
import pandas as pd
import warnings
warnings.filterwarnings("ignore")  # 忽略某些不影响程序的提示
import pyLDAvis
import pyLDAvis.gensim
import os
from pathlib import Path

class Topic(object):
    def __init__(self, cwd):
        """
        初始化
        :param cwd: 当前代码所在的文件夹路径
        """
        self.output = Path(cwd).joinpath('output')
        self.output.mkdir(exist_ok=True)


        bestmodelspath = Path(cwd).joinpath('output', 'model')
        bestmodelspath.mkdir(exist_ok=True)



    def create_dictionary(self, documents, no_below=20, no_above=0.5):
        """
        输入带documents构建词典dictionary (同时会默认将词典命名为dictionary.dict存储到output文件内)
        :param documents: 列表； 注意documents中的每个document是词语列表
        :param no_below: 整数；构建词典空间时，词语出现次数低于no_below的词语剔除掉
        :param no_above: 小数范围(0, 1)； 构建词典空间时，词语词频比高于no_above的词语剔除掉。
        :return: 返回corpus
        """
        self.documents = self.__add_bigram(documents=documents, min_count=no_above)
        self.dictionary = Dictionary(self.documents)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        fpath = str(self.output.joinpath('dictionary.dict'))
        self.dictionary.save(fpath)

    def load_dictionary(self, dictpath='output/dictionary.dict'):
        """
        导入词典(同时在此步骤可自动构建corpus)
        :param dictpath: 词典文件的路径
        :return: 返回corpus
        """

        self.dictionary = Dictionary.load(dictpath)

    def create_corpus(self, documents):
        """
        输入带documents构建corpus;
        :param documents: 列表； 注意documents中的每个document是词语列表
        :return: 返回corpus
        """
        self.corpus = [self.dictionary.doc2bow(document) for document in documents]
        self.documents = documents

    def __add_bigram(self, documents, min_count):
        """
        分词容易把一个意思的词组分成多个词，考虑词组整体作为一个整体。
        :param documents: 文档集(语料库)
        :param min_count: 词组出现次数少于于min_count，该词组不会被加到documents中
        :return: 更新后的documents
        """
        bigram = gensim.models.Phrases(documents, min_count=min_count)
        for idx in range(len(documents)):
            for token in bigram[documents[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    documents[idx].append(token)
        return documents



    def train_lda_model(self, n_topics, fname='lda.model', epochs=20, iterations=300):
        """
        训练lda话题模型，运行时间较长，请耐心等待~
        :param n_topics:  指定的话题数
        :param fname:  模型的文件名（用来保存训练好的lda模型）,默认存放在output文件夹内
        :param epochs: 使用数据集训练的轮数。epochs越大，训练越慢
        :param iterations:  对每个document迭代(学习)的次数；iterations越大，训练越慢
        :return:  返回训练好的lda模型
        """
        self.model = gensim.models.LdaMulticore(corpus=self.corpus,
                                                num_topics=n_topics,
                                                id2word=self.dictionary,
                                                workers=4,
                                                chunksize=1000,
                                                eval_every=None,
                                                passes=epochs,
                                                iterations=iterations,
                                                batch=True)
        fpath = str(self.output.joinpath('model', fname))
        self.model.save(fpath)
        return self.model

    def load_lda_model(self, modelpath='output/model/lda.model'):
        """
        导入之前训练好的lda模型
        :param modelpath: lda模型的文件路径 (存放在output中的best_model和models文件夹中内)
        :return:
        """
        self.model = gensim.models.LdaModel.load(modelpath, mmap='r')

    def show_topics(self, formatted=True):
        """
        显示话题与对应的特征词之间的权重关系
        :param formatted: 特征词与权重是否以字符串形式显示
        :return: 列表
        """
        return self.model.show_topics(formatted=formatted)

    def visualize_lda(self, fname='vis.html'):
        """
        可视化LDA模型。如果notebook中无法看到可视化效果，请在output文件夹内找到 fname所对应的html文件，用浏览器打开观看
        :param fname: 可视化html文件，默认存放在output文件夹内，运行结束后找到vis.html并用浏览器打开
        :return:
        """
        vis = pyLDAvis.gensim.prepare(self.model, self.corpus, self.dictionary)
        fpath = str(self.output.joinpath(fname))
        pyLDAvis.save_html(vis, fpath)



    def get_document_topics(self, document):
        """
        :param document: 词语列表
        :return:
        """
        return self.model.get_document_topics(self.dictionary.doc2bow(document))

    def topic_distribution(self, raw_documents, fname='文本话题归类.csv'):
        """
        将raw_documents与对应的所属话题一一对应，结果以fname的csv文件存储。返回raw_documents话题分布情况
        :param raw_documents: 列表；原始的文档数据集
        :param fname: csv文件名, 默认存放在output文件夹内
        :return:
        """
        topics = []
        for document in self.corpus:
            topic_probs = self.model.get_document_topics(document)
            topic = sorted(topic_probs, key=lambda k: k[1], reverse=True)[0][0]
            topics.append(topic)
        df = pd.DataFrame({'topic': topics, 'text': raw_documents})
        fpath = str(self.output.joinpath(fname))
        df.to_csv(fpath, mode='w', encoding='utf-8', index=False)
        return df['topic'].value_counts()




