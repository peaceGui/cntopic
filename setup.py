from setuptools import setup
import setuptools

setup(
    name='cntopic',     # 包名字
    version='0.1',   # 包版本
    description='中文文本话题分析',   # 简单描述
    author='大邓',  # 作者
    author_email='thunderhit@qq.com',  # 邮箱
    url='https://github.com/thunderhit/cntopic',      # 包的主页
    packages=setuptools.find_packages(),
    install_requires=['gensim>=3.8.3', 'jieba', 'pyldavis>=2.1.2', 'pandas'],
    python_requires='>=3.5',
    license="MIT",
    keywords=[
        'topic model',
        'LDA',
        'topic model',
        '支持中文',
        '话题模型',
        'text mining',
        '中文文本分析',
        '文本挖掘'
    ],
    long_description=open('README.md').read(), # 读取的Readme文档内容
    long_description_content_type="text/markdown")  # 指定包文档格式为markdown

