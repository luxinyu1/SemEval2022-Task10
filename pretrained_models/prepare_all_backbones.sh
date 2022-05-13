cd pretrained_models
wget -O "ernie_2.0_skep_large_en.tar.gz" --no-check-certificate https://senta.bj.bcebos.com/skep/ernie_2.0_skep_large_en.tar.gz
tar -xzvf ernie_2.0_skep_large_en.tar.gz
rm ernie_2.0_skep_large_en.tar.gz
cd ..
pip install paddlepaddle==1.8.5
bash ./scripts/convert_paddle_model_to_pytorch.sh
cd pretrained_models
git lfs install
git clone https://huggingface.co/roberta-large/
git clone https://huggingface.co/xlm-roberta-large/
git clone https://huggingface.co/sentence-transformers/LaBSE
git clone https://huggingface.co/bert-large-uncased/
git clone https://huggingface.co/deepset/roberta-large-squad2/
git clone https://huggingface.co/deepset/xlm-roberta-large-squad2/
git clone https://huggingface.co/NbAiLab/nb-bert-base/
git clone https://huggingface.co/bert-base-multilingual-cased/
