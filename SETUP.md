# Recipe Summarization

## Server setup

### multimedia host:

SSH connection:
```
ssh jose.mena@172.20.120.66
cd ~/dev/recipe-summarization
source ~/dev/envs/recipe-summarization/bin/activate
``` 
## Modifications

* requirements

add rouge==0.3.1
for local, remove the gpu suffix of the tensorflow dependency

* vocabulary-embedding.py 

embedding_dim set to 300

* fasttext-vocabulary-embedding.py

Create a new file to read FastText embeddings.
Get the FastText embeddings:

```
cd data
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
```



