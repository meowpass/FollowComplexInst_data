Implementation and corresponding data of the paper "From Complex to Simple: Enhancing Multi-Constraint Complex Instruction Following Ability of Large Language Models".
Since the public GitHub URL will be revealed after ARR procedure, this repository will be removed soon.

## ⚙️How to Use the Code

### Install Dependencies

```
conda create -n fcs python=3.10.9
conda activate fcs
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Obtain Complex Instructions

#### Complex Instruction Synthesis

To obtain complex instructions: 
- First, we collect seed instructions from three widely used instruction-tuning datasets. 
- Then, we rewrite the instructions to incorporate multiple constraints. 
Here, you can complete the whole procedure by running the script `gen_inst.sh`:

```shell
python ../get_data/gen_inst.py \
    --seed_path=../get_data/data/seed_data.jsonl  \
    --data_path=../get_data/data/data.jsonl\
    --api_key=YOUR_API_KEY_TO_ACESS_GPT4\
```

An example of complex instrutcion is shown as below:
![image](https://github.com/meowpass/FollowComplexInstruction/assets/56729976/cd6810af-d472-42e7-afff-43b83e30dc42)
Here are 3 different constraints in the instructions.

#### Get Model Outputs

You need to do inference with your model to get the responses to the complex instructions. Here, we provide a script to do inference for LLaMA via the script `do_inference.sh`:

```shell
CUDA_VISIBLE_DEVICES=YOUR_CUDA_DEVICES python ../get_data/do_inference.py \
    --data_path=../get_data/data/data.jsonl \
    --res_path=../get_data/data/res_llama2.jsonl \
    --model_path=PATH_TO_YOUR_MODEL\
    --lora_path=PATH_TO_YOUR_LORA_WEIGHT IF YOU USE LORA \
```



#### Teacher Correction

We propose a discrimination-based approach for obtaining the output, shown to be more effective than directly generating output with advanced LLMs. 

First, we utilize the test scripts from [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval) to identify the constraints the model failed to follow since the constraints are objective and automatically verifiable. Simply run the script `check.sh`:

```shell
python ../get_data/check.py \
    --input_data=../get_data/data/data.jsonl \
    --input_response_data=../get_data/data/res_llama2.jsonl \
    --output_dir=../get_data/data/ \
    --output_file_name=checked_res_llama2
```

Then, we adopt advanced LLMs (teacher model) GPT-3.5-turbo to correct the failed constraints one by one. You can correct the response to simultaneously get data for IFT and DPO with the script `correct.sh`:

```shell
python ../get_data/correct.py \
    --res_path=../get_data/data/res_llama2.jsonl  \
    --ift_data_path=../dpo_train/data/ift_train.jsonl \
    --dpo_data_path=../dpo_train/data/dpo_train.jsonl \
    --api_key=YOUR_API_KEY_TO_ACESS_GPT4\
```


### Contrastive Method (Go for DPO Training)
The slight changes in the instruction (i.e. `json` to `xml`) can cause substantial output differences. Hence, negative samples failing to meet certain constraints, also offer valuable supervision signals. we leverage the positive and negative samples through reinforcement learning fine-tuning.

Here, we provide a revised implementation for an advanced DPO in `dpo_train`. You can set your model_path and data_path in `dpo_train/dpo_train.py`. Then, you can train the model with the script `train_dpo.sh`:

```shell
CUDA_VISIBLE_DEVICES=YOUR_CUDA_DEVICES accelerate launch \
    --config_file ../dpo_train/deepspeed_zero1.yaml dpo_train.py \
    --output_dir=PATH_TO_SAVE_MODEL \
```
