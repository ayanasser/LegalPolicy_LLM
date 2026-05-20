tensorboard --logdir runs/qlora-qwen2.5-1.5b-v1/runs --port 6006

bash scripts/run_chat_ui.sh  #finetuned 3b knowledge 
bash scripts/finetuned_3b_knowledge_run_chat_ui.sh
