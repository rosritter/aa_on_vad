# aa_on_vad
adversarial attacks on vocie activity detection


### Papper

1. [paper 1](https://arxiv.org/pdf/2103.03529v1)


### git

1. [rep 1](@NickWilkinson37/voxseg)

### dataset
1. [ava speech](https://research.google.com/ava/download.html#ava_speech_download)
2. [hf dipco](https://huggingface.co/datasets/huckiyang/DiPCo)
3. [kggl lazyrac00n](https://www.kaggle.com/datasets/lazyrac00n/speech-activity-detection-datasets?select=Data)
4. [librispeech](https://pytorch.org/audio/2.5.0/generated/torchaudio.datasets.LIBRISPEECH.html#torchaudio.datasets.LIBRISPEECH)

1. [urban-sound](https://timeseriesclassification.com/description.php?Dataset=UrbanSound)
2. [musan](https://www.openslr.org/17/)
3. [esc-50](https://github.com/karolpiczak/ESC-50)

next step trying to learn model



### Env. prep.

`export PYTHONPATH=$PWD:$PATH`

### Download datasets
`!make -f datasets/Makefile all`
