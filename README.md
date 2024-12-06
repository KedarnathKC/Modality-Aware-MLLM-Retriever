# Modality Aware MLLM Retriever

### Requirements
```angular2html
torch==2.2.0
torchmetrics==1.3.1
torchvision==0.17.0
transformers==4.46.3
evaluate==0.4.1
numpy==1.24.4
re==2.2.1
tensorboard==2.16.2
pyyaml==6.0
json==2.0.9
seaborn==0.12.2
matplotlib==3.7.1
pandas==1.5.3
PIL==10.0.1
tqdm==4.65.0
p-tqdm==1.2
scipy==1.11.4
networkx==3.1
```

To start the fine-tuning and/or evaluation and/or retrieval actions, run the below commands.

<hr>

```bash 
python3 main.py 
```

<hr>

### Dataset configuration
- Update the Common.DataSet.Path manually or add environment variable ${DATASET_PATH}
- Use FilterDomains to choose which dataset to use: visualnews, fashion200k and mscoco 
- Add/Remove retrival scores to calculate using Metrics.Name

<hr>

### Fine-Tuning configuration
-  FineTuning.Action: True 
- Set UseModalityNegatives to true use mined negatives, otherwise use random negatives. 
  - When set to true, update ModalityNegativesPath to file containing mined negatives for queries.
- CandidateSize controls number of negative candidates to train when using random inbatch or mined negatives. 

<hr style="height: 0.5px">

### Evaluation configuration
- Evaluation.Action: True

<hr style="height: 0.5px">

### Retrieval configuration
- Retrieval.Action: True

<hr style="height: 0.5px">

### Results structure

    .
    ├──...     
    ├── Results                
    │    ├── FineTuned                
    │    │   ├── <Model Name>                      
    │    │   │    ├── <DataSet>                   
    │    │   │    │    ├── run_{index}                  
    │    │   │    │    │    ├── logs
    │    │   │    │    │    ├── training
    │    │   │    │    │    ├── tuned-model
    │    ├── Evaluation                
    │    │    ├── <Model Name>                      
    │    │    │    ├── <DataSet>                   
    │    │    │    │    ├── run_{index}                  
    │    │    │    │    │    ├── local
    │    │    │    │    │    ├── embeddings
    │    │    │    │    │    ├── <DataSet>.index

