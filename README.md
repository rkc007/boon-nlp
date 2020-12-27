# BOON : An Intelligent Chatbot using Deep Neural Network

## Description
 - Developed chatbot using encoder and decoder based Sequence-to-Sequence (Seq2Seq) model from Google’s Neural Machine Translation (NMT) module and Cornell Movie Subtitle Corpus.
 - Seq2Seq architecture built on Recurrent Neural Network and was optimized with bidirectional LSTM cells.
 - Enhanced chatbot performance by applying Neural Attention Mechanism and Beam Search.
 - Attained testing perplexity of 46.82 and Bleu 10.6.
 - Developed backend using Python and front-end using Python and PyQT.
 
 ## Sample Conversations: 
 <kbd>
    <img src=https://github.com/rkc007/boon-nlp/blob/main/images/chat_gen.png>
 </kbd>
 
  ## GUI: 
 <kbd>
    <img src=https://github.com/rkc007/boon-nlp/blob/main/images/chat_gui.png>
 </kbd>

 ### Requirement: 
- Google's Tensorflow 1.6 
- Python 3.4
- tensorboard

### Instruction For running chatbot:
- Activate tensorflow  
 `source ~/tensorflow/bin/activate`  
- Move to folder "project/code/"  
   `cd /home/rkc007/projects/dode/`
-  `python chat_gui.py`

### Training

1. Activate tensorflow
   > source ~/tensorflow/bin/activate 
2. move to nmt module:
   change prefix "/home/rkc007/project" to current project location:
   > cd /home/rkc007/project/code/nmt
3. remove model folder: 
   change prefix "/home/rkc007/project" to current project location  
   > rm -r /home/rkc007/project/code/inout/output/nmt_model
4. create nmt training output folder:
   change prefix "/home/rkc007/project" to current project location
   > mkdir /home/rkc007/code/inout/output/nmt_model
   
5. update location of "vocab_prefix", "train_prefix", "dev_prefix", "test_prefix", "out_dir" by changing the prefix in all these files with  where prefix "/home/rkc007/project" should be changed to current project folder. Also, have option to change - "num_train_steps", "steps_per_stats", "num_units", "learning_rate", "beam_width", "num_translations_per_input"

### Following is the training instruction :

```
python -m nmt.nmt \
    --src=vi --tgt=en \
    --vocab_prefix=/home/rkc007/project/Code/inout/input/nmt_data/vocab  \
    --train_prefix=/home/rkc007/project/code/inout/input/nmt_data/train \
    --dev_prefix=/home/rkc007/project/code/inout/input/nmt_data/tst2012  \
    --test_prefix=/home/rck007/project/code/inout/input/nmt_data/tst2013 \
    --out_dir=/home/rkc007/project/code/inout/output/nmt_model \
    --attention=scaled_luong \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=512 \
    --learning_rate=0.001 \
    --decay_steps=1 \
    --start_decay_step=1 \
    --beam_width=10 \
    --length_penalty_weight=1.0 \
    --optimizer=adam \
    --encoder_type=bi \
    --num_translations_per_input=30
```

### For viewing training performance: 

1. Activate tensorflow   
   `source ~/tensorflow/bin/activate` 

2. move to nmt module:   
   `cd /home/rkc007/project/code/nmt`
3. enter following command  
   `tensorboard --port 22222 --logdir /home/rkc007/project/code/inout/output/nmt_model` 
   
### Future Work: 
1. Working on creating a server using Flask to host this chatbot.
2. Chatbot to work on public domain with different languages

### Acknowledgements:
Major Thanks to *[Dr. Kevin Scannell ](https://cs.slu.edu/~scannell/index.html)* for teaching NLP course.   
*[May Li](https://github.com/mayli10/deep-learning-chatbot)*   
*[TensorLayer Community](https://github.com/tensorlayer/seq2seq-chatbot)*    
*[Aditya Kumar](https://github.com/adi2381/ai-chatbot)* for GUI   


