DATASET_DIR = datasets


# ESC (noise)
# github.com/karolpiczak/ESC-50
ESC_LINK = github.com/karolpiczak/ESC-50#download

# Musan mirrors	
# Summary: A corpus of music, speech, and noise
# openslr.org/17/
# https://us.openslr.org/resources/17/musan.tar.gz
# https://openslr.elda.org/resources/17/musan.tar.gz
# https://openslr.magicdatatech.com/resources/17/musan.tar.gz

MUSAN_LINK = https://www.openslr.org/resources/17/musan.tar.gz


# URBAN_SOUND 
# https://timeseriesclassification.com/description.php?Dataset=UrbanSound
URBAN_SOUND_LINK = https://timeseriesclassification.com/aeon-toolkit/UrbanSound.zip

all: esc50 musan urbansound librispeech

esc50:
	wget -P $(DATASET_DIR) $(ESC_LINK) 
	unzip $(DATASET_DIR)/master.zip -d $(DATASET_DIR)


musan:
	wget -P $(DATASET_DIR) $(MUSAN_LINK)
	tar -xzvf $(DATASET_DIR)/musan.tar.gz -C $(DATASET_DIR)

urbansound:
	wget -P $(DATASET_DIR) $(URBAN_SOUND_LINK) 
	unzip $(DATASET_DIR)/UrbanSound.zip -d $(DATASET_DIR)

librispeech:
	python $(DATASET_DIR)/torchlibri/download_torch_libri_ds.py
