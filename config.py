
import os
import torch


class _Config:
    
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    
    JARVIS_CSV          = "jarvis_properties.csv"

    
    TEXTS_LEGACY_JSONL  = "jarvis_aligned_texts.jsonl"     
    TEXTS_CLEAN_JSONL   = "jarvis_clean_texts.jsonl"       

    
    EMBED_DIR_LEGACY    = "text_embeddings"
    EMBED_DIR_CLEAN     = "text_embeddings_clean"

    
    SPLIT_DIR           = "dataset_split"
    TRAIN_JSONL_TPL     = "{split_dir}/train/train_metadata.jsonl"
    VAL_JSONL_TPL       = "{split_dir}/val/val_metadata.jsonl"
    TEST_JSONL_TPL      = "{split_dir}/test/test_metadata.jsonl"

    
    CKPT_MULTIMODAL  = "checkpoints_multimodal"                    
    CKPT_GRAPH_ONLY     = "checkpoints_graphonly"
    RESULTS_DIR         = "results"

    
    LLM_LOCAL_PATH      = os.environ.get("CLAMP_LLM_PATH", "Qwen/Qwen3-8B")

    
    IN_NODE_DIM   = 10
    HIDDEN_DIM    = 256
    EMBED_DIM     = 4096
    N_LAYERS      = 4
    NUM_RBF       = 16
    CUTOFF        = 8.0
    LLM_DIM       = 4096
    DROPOUT       = 0.3

    
    BATCH_SIZE    = 64
    EPOCHS        = 150
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY  = 5e-4
    WARMUP_EPOCHS = 5
    MIN_LR        = 1e-6
    GRAD_CLIP     = 1.0

    
    TEMPERATURE       = 0.07
    LAMBDA_REG        = 1.0
    LAMBDA_NCE_INIT   = 1.0
    LAMBDA_NCE_FINAL  = 0.1
    NCE_DECAY_START   = 0.3   # epoch fraction
    NCE_DECAY_END     = 0.8

    
    EHULL_STABLE_THRESHOLD = 0.05
    EHULL_STABLE_WEIGHT    = 3.0

    
    NUM_WORKERS = 4
    PIN_MEMORY  = True

    
    EMBED_BATCH_SIZE = 128
    EMBED_MAX_LEN    = 256

    
    MAX_TEST_SAMPLES   = 10000
    TOP_K_RAG          = 10
    TEMPERATURE_RAG    = 0.05

    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    PROP_NAMES = ['bandgap', 'formation_energy', 'ehull']
    PROP_UNITS = ['eV', 'eV/atom', 'eV/atom']

    
    def update_for_quick_test(self):
        self.EPOCHS    = 5
        self.BATCH_SIZE = 16
        self.MAX_TEST_SAMPLES = 200
        print('Quick-test mode enabled: epochs=5, batch_size=16, max_test_samples=200')

    def __repr__(self):
        attrs = [a for a in dir(self) if not a.startswith('_') and a.isupper()]
        return "Config(\n" + "\n".join(
            f"  {a}: {getattr(self, a)}" for a in attrs
        ) + "\n)"



CFG = _Config()


if __name__ == "__main__":
    print(CFG)
