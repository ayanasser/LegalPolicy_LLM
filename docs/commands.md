evaluation:

cd /home/aya/master/genai/LegalPolicy_LLM
conda activate legalpolicy
python -m apps.unified_ui.app


PYTHONPATH=src python -m apps.unified_ui.app
OR

./scripts/run_unified_ui.sh

 python scripts/eval_rag.py --system bilingual-rag --judge ollama --phase all



tensorboard --logdir runs/qlora-qwen2.5-1.5b-v1/runs --port 6006

bash scripts/run_chat_ui.sh  #finetuned 3b knowledge 
bash scripts/finetuned_3b_knowledge_run_chat_ui.sh
