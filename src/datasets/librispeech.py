import torchaudio

'''
https://pytorch.org/audio/main/generated/torchaudio.datasets.LIBRISPEECH.html#torchaudio.datasets.LIBRISPEECH
'''
def get_librispeech_example():
    ds = torchaudio.datasets.LIBRISPEECH(root='datasets/torchlibri',
                                     url ='test-clean',
                                     download=True)
    return ds