# spoken_qa

## Dataset

Spoken questions automatically generated via Google Translate API and then transcribed with the Facebook wav2vec2 large ASR model [can be downloaded here](https://drive.google.com/drive/folders/1c3QByuZROJbJZzEXEOKOgpHCtU9gH-s9?usp=sharing)

* Simple Questions https://github.com/askplatypus/wikidata-simplequestions
* Natural Questions https://ir-datasets.com/beir.html#beir/nq
* MS MARCO https://ir-datasets.com/beir.html#beir/msmarco/dev

Speech generation script: https://github.com/svakulenk0/spoken_qa/blob/main/notebooks/generate_speech.ipynb


```
cd data
wget https://raw.githubusercontent.com/askplatypus/wikidata-simplequestions/master/annotated_wd_data_train.txt
wget https://raw.githubusercontent.com/askplatypus/wikidata-simplequestions/master/annotated_wd_data_valid.txt
```
